from phas.client.api import Client, SamplingROITask, Slide
from phas.gcs_handler import GoogleCloudOpenSlideWrapper, SelfManagedMultiFilePageCache
from phas.dltrain import compute_sampling_roi_bounding_box, affine_transform_roi, draw_sampling_roi
from google.cloud import storage
from tangle_cnn.wildcat_main import TrainedWildcat
from tangle_cnn.osl_worker import OpenSlideHistologyDataSource
import openslide
import pandas as pd
import numpy as np
import os
import json
import time
import SimpleITK as sitk
from PIL import Image   
import sys

import pymetis
from scipy.sparse import csr_matrix, eye
from sklearn.feature_extraction.image import grid_to_graph
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

import yaml

from pydantic import BaseModel
from typing import Dict, Literal, List

def crop_region(region_to_crop, region_to_crop_to):
    """ Crop `image_region` to fit within `crop_region`.

    Args:
        region_to_crop: tuple (x_index, y_index, x_size, y_size)
            Defines the region to be cropped.
        region_to_crop_to: tuple (x_index, y_index, x_size, y_size)
            Defines the cropping boundaries.

    Returns: Cropped region as a tuple (new_x_index, new_y_index, new_x_size, new_y_size), 
        or None if the regions do not overlap and cropping cannot be done.
    """
    # Extract initial region indices and sizes
    ix, iy, ix_size, iy_size = region_to_crop
    cx, cy, cx_size, cy_size = region_to_crop_to

    # Check if the regions overlap; if not, return None
    if ix >= cx + cx_size or ix + ix_size <= cx or iy >= cy + cy_size or iy + iy_size <= cy:
        return None  # No cropping possible, regions do not overlap

    # Adjust the x index and size
    if ix < cx:
        ix_size -= cx - ix
        ix = cx
    if ix + ix_size > cx + cx_size:
        ix_size = (cx + cx_size) - ix

    # Adjust the y index and size
    if iy < cy:
        iy_size -= cy - iy
        iy = cy
    if iy + iy_size > cy + cy_size:
        iy_size = (cy + cy_size) - iy

    return (ix, iy, ix_size, iy_size)


def crop_region_by_extents(region_to_crop, region_to_crop_to):
    """Crop a region specified as (xmin,ymin,xmax,ymax)."""
    ix0, iy0, ix1, iy1 = region_to_crop
    cx0, cy0, cx1, cy1 = region_to_crop_to
    crop = crop_region((ix0, iy0, ix1-ix0, iy1-iy0), (cx0, cy0, cx1-cx0, cy1-cy0))
    if crop:
        ox0, oy0, sx, sy = crop
        return (ox0, oy0, ox0+sx, oy0+sy) 
    else:
        return None


class WildcatInferenceParameters(BaseModel):
    """Parameters for WildCat inference"""
    
    #: Name of the stain
    stain: str
    
    #: Suffix for the output files, may include name of experiment, etc.
    suffix: str
    
    #: Contrasts to evaluate, in form 'name':class_id, where class_id is one of the classes in WildCat training
    contrasts: Dict [str, int]
    
    #: Resolution in mm to which slides should be resampled before running Wildcat.
    target_resolution: float = 0.0004
    
    #: Additional downsampling for the density maps - to conserve space
    downsample: int = 8
    
    #: Window size for inference
    window: int = 1024
    
    #: Normalization for computing summary statistics, 'minus_others' means class K minus all other classes, 'none' means just class K
    normalization: Literal ['none','minus_others'] = 'none'
    
    #: Quantiles to report from cluster-level activation map averages
    quantiles: List [float] = [ 0.5, 0.8, 0.9, 0.95, 0.98, 0.99 ]
    
    #: Area of a cluster for cluster-level statistics
    cluster_area: float = 0.04
    
    #: Additional padding for sampling ROIs
    roi_padding: int = 0


def make_disk(spacing, radius, padding=4):
    n = int(padding + np.ceil(radius / spacing))
    t = np.linspace(-n*spacing, n*spacing, 2*n+1)
    tt = np.tile(t**2, [2*n+1,1])
    return np.where(tt + tt.T < radius**2, 1.0, 0.0)


class WildcatInferenceOnSamplingROI:

    def __init__(self, wildcat:TrainedWildcat, param: WildcatInferenceParameters):
        self.wildcat = wildcat
        self.param = param
        self.gcs_client = None
        self.gcs_cache = None
        
    def _wdir(self, rootdir, slide:Slide):
        specimen = slide.specimen_private
        stain = slide.stain
        return os.path.join(rootdir, specimen, f'{specimen}_{stain}_slide_{slide.slide_id}')
        
    def _fn_state(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_state.json')
    
    def _fn_roi_base(self, workdir, slide_id, roi):
        return os.path.join(workdir, f'slide_{slide_id}_roi_{roi["id"]}')

    def _fn_density(self, workdir, slide_id, roi):
        return self._fn_roi_base(workdir, slide_id, roi) + f'_density_{self.param.suffix}.nii.gz'
    
    def _fn_mask(self, workdir, slide_id, roi):
        return self._fn_roi_base(workdir, slide_id, roi) + f'_mask.nii.gz'
    
    def _fn_rgb(self, workdir, slide_id, roi):
        return self._fn_roi_base(workdir, slide_id, roi) + f'_rgb.nii.gz'
    
    def _fn_cluster_labels(self, workdir, slide_id, roi):
        return self._fn_roi_base(workdir, slide_id, roi) + f'_density_{self.param.suffix}_clusters.nii.gz'
        
    def _fn_cluster_heatmap(self, workdir, slide_id, roi, contrast):
        return self._fn_roi_base(workdir, slide_id, roi) + f'_density_{self.param.suffix}_clustermap_{contrast}.nii.gz'
    
    def _fn_sampling_roi_stats(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_stats_{self.param.suffix}.json')
        
    def _fn_sampling_roi_cluster_stats(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_cluster_activation_{self.param.suffix}.csv')
    
    def _fn_sampling_roi_local_max_stats(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_local_max_activation_{self.param.suffix}.csv')
        
    def compute_superpixel_stats(self, work_dir, slide_id):
        
        # Look in the slide directory for the listing of ROIs
        with open(self._fn_state(work_dir, slide_id)) as sroi_json_fd:
            rois = json.load(sroi_json_fd)['rois']

        # Get the list of unique labels
        unique_labels = set([x['label'] for x in rois])

        # Create a pandas frame to hold individual patch measures for validation
        patch_data = {'id':[], 'slide':[], 'label':[], 'patch':[], 'contrast':[], 'value':[], 'ncells':[]}

        # Repeat for each label
        quantiles = dict()
        for label in unique_labels:

            # List of tangle/thread values for all cells
            val = { k:[] for k in self.param.contrasts.keys() }
            n_cells = 0
            for i, roi_dict in enumerate([x for x in rois if x['label'] == label]):
                
                # The density image may not exist because the sampling region may be outside 
                # of the slide
                fn_den = self._fn_density(work_dir, slide_id, roi_dict)
                if not os.path.exists(fn_den):
                    continue

                # Load the density image and mask image
                img_density = sitk.ReadImage(fn_den)

                # Load the mask image for this ROI
                img_roi = sitk.ReadImage(self._fn_mask(work_dir, slide_id, roi_dict))

                # We want the average area of a sampling region to be 200x200 microns
                pix_area = img_density.GetSpacing()[0] * img_density.GetSpacing()[1]
                pix_per_cluster = self.param.cluster_area / pix_area

                # Get the density maps for the tangles and threads (or whatever we are looking for)
                density = sitk.GetArrayFromImage(img_density)
                if self.param.normalization == 'minus_others':
                    d = { k: np.maximum(density[:,:,v] * 2 - density.sum(axis=2), 0) for k,v in self.param.contrasts.items() }
                else:
                    d = { k: np.maximum(density[:,:,v],0) for k,v in self.param.contrasts.items() }
                    # d = { k: density[:,:,v] > 0 for k,v in self.param.contrasts.items() }

                # Resample the boxes into the density space
                img_roi_big = sitk.Resample(img_roi, img_density, interpolator=sitk.sitkNearestNeighbor)
                roi = sitk.GetArrayFromImage(img_roi_big).astype(np.float32)

                # Find the number of averaging patches to generate
                n_clusters = int(np.ceil(np.sum(roi > 0) / pix_per_cluster))
                if n_clusters == 0:
                    continue

                n_cells += n_clusters
            
                # Generate the clusters with METIS
                G = grid_to_graph(roi.shape[0], roi.shape[1], 1, mask=roi>0, return_as=csr_matrix)
                G = G - eye(G.shape[0])
                Gp = pymetis.part_graph(n_clusters, xadj=G.indptr, adjncy=G.indices, recursive=True)

                # Label the pixels in the ROI by METIS part number
                roi_part = np.zeros_like(roi, dtype=np.int32)
                roi_part[roi>0] = np.array(Gp[1], dtype=np.int32)+1
                
                # Save the clusters
                img_roi_part = sitk.GetImageFromArray(roi_part, isVector=False)
                img_roi_part.CopyInformation(img_density)
                sitk.WriteImage(img_roi_part, self._fn_cluster_labels(work_dir, slide_id, roi_dict))

                # Compute the average tangles/threads in each ROI
                for k, d_roi_k in d.items():
                    # Create an image for this contrast where we assign each ROI the computed value
                    roi_hmap = np.zeros_like(roi, dtype=np.float32)
                    roi_hmap[roi==0] = np.nan
                    
                    # Use bincount to quickly integrate over patches for this computation
                    bc = np.bincount(roi_part.flatten(), weights=np.maximum(d_roi_k, 0).flatten()) / np.bincount(roi_part.flatten())
                    bc[0] = 0
                    roi_hmap[:] = bc[roi_part[:]]
                    val[k] = val[k] + bc[1:].tolist()

                    for key in ('id','slide','label'):
                        patch_data[key] += [roi_dict[key]] * (len(bc)-1)
                    patch_data['patch'] += range(0, len(bc)-1)
                    patch_data['contrast'] += [k] * (len(bc)-1)
                    patch_data['value'] += bc[1:].tolist()
                    
                    # Quickly count how many discrete cells there are - this is lame because we are doing watershed
                    # on downsampled images, but this is just a test
                    image_ws = d_roi_k
                    coords = peak_local_max(image_ws, footprint=np.ones((9, 9)), labels=image_ws > 0)
                    zmask = np.zeros(image_ws.shape, dtype=bool)
                    zmask[tuple(coords.T)] = True
                    markers, _ = ndi.label(zmask)
                    labels = watershed(image_ws, markers, mask=image_ws>0)
                    
                    # Count how many unique pixels there are per ROI
                    idx = labels > 0
                    a, b = np.unique(np.stack([roi_part[idx], labels[idx]]),axis=1)
                    df_count = pd.DataFrame({'a':a, 'b':b}).groupby('a').size().reset_index(name='n')
                    df_count = df_count.merge(pd.DataFrame({'a':np.arange(np.max(roi_part)+1)}), how='right')
                    df_count = df_count.replace({np.nan:0})
                    patch_data['ncells'] += (np.array(df_count.n, dtype=int)[1:]).tolist()

                    # Save the ROI heat map
                    img_hmap = sitk.GetImageFromArray(roi_hmap, isVector=False)
                    img_hmap.CopyInformation(img_density)
                    sitk.WriteImage(img_hmap, self._fn_cluster_heatmap(work_dir, slide_id, roi_dict, k))

            # Compute the quantiles
            if n_cells > 0:
                quantiles[label] = { 'n_cells': n_cells }
                for k,v in val.items():
                    quantiles[label][k] = { q : np.quantile(v, q) for q in self.param.quantiles }

        # Write the summary statistics
        with open(self._fn_sampling_roi_stats(work_dir, slide_id), 'wt') as stats_fd:
            json.dump(quantiles, stats_fd)

        # Write the per-patch data
        df = pd.DataFrame(data=patch_data)
        df.to_csv(self._fn_sampling_roi_cluster_stats(work_dir, slide_id))
    
    def compute_multiscale_mass_stats(self, work_dir, slide_id):
        
        # Look in the slide directory for the listing of ROIs
        with open(self._fn_state(work_dir, slide_id)) as sroi_json_fd:
            rois = json.load(sroi_json_fd)['rois']

        # Get the list of unique labels
        unique_labels = set([x['label'] for x in rois])

        # Dataframe with measures at different scales
        patch_data = {'id':[], 'slide':[], 'label':[], 'contrast':[], 'scale': [], 'radius':[], 'local_max':[]}

        # Scales are in units of 1/mm, i.e., scale of 100 means 10µm radius, scale of 2 means 500µm
        # radius, and scale of 0 means all pixels in the image. Larger scale means more local
        scales = [0, 1, 2, 5, 10, 20, 50, 100]
        
        # Kernels will be generated on the fly based on the spacing of the density image
        kernels = {}

        # Repeat for each label
        for label in unique_labels:

            # List of tangle/thread values for all cells
            val = { k:[] for k in self.param.contrasts.keys() }
            n_cells = 0
            for i, roi_dict in enumerate([x for x in rois if x['label'] == label]):
                
                # The density image may not exist because the sampling region may be outside 
                # of the slide
                fn_den = self._fn_density(work_dir, slide_id, roi_dict)
                if not os.path.exists(fn_den):
                    continue

                # Load the density image and mask image
                img_density = sitk.ReadImage(fn_den)

                # Load the mask image for this ROI
                img_roi = sitk.ReadImage(self._fn_mask(work_dir, slide_id, roi_dict))
                                
                # Get the density maps for the tangles and threads (or whatever we are looking for)
                density = sitk.GetArrayFromImage(img_density)
                if self.param.normalization == 'minus_others':
                    d = { k: np.maximum(density[:,:,v] * 2 - density.sum(axis=2), 0) for k,v in self.param.contrasts.items() }
                else:
                    d = { k: np.maximum(density[:,:,v],0) for k,v in self.param.contrasts.items() }
                    
                # Resample the boxes into the density space
                img_roi_big = sitk.Resample(img_roi, img_density, interpolator=sitk.sitkNearestNeighbor)
                roi = sitk.GetArrayFromImage(img_roi_big).astype(np.float32)

                # Compute the kernels for this density image spacing 
                spacing = img_density.GetSpacing()[0]
                if spacing not in kernels:
                    # For each scale, generate the corresponding kernel. We can reuse the kernels from call to call
                    kernels[spacing] = { s: make_disk(spacing, 1.0 / s) if s > 0 else None for s in scales }
                        
                # Convolve the masked thresholded density map with the disks
                for scale, kernel in kernels[spacing].items():

                    # Convolve mask with kernel
                    roi_mf = fftconvolve(roi, kernel, mode='same')[roi > 0] if kernel is not None else None

                    # Compute the average tangles/threads in each ROI
                    for k, d_roi_k in d.items():
                    
                        # Compute the maximum local mean of the masked region or ROI-level mean
                        d_bin = np.where(d_roi_k > 0, 1, 0)
                        d_thresh = d_bin * d_roi_k
                        if scale > 0:
                            d_thresh_mf = fftconvolve(d_thresh * roi, kernel, mode='same')
                            d_local_max = np.max(d_thresh_mf[roi > 0] / roi_mf)
                        else:
                            d_local_max = np.sum(d_thresh * roi) / np.sum(roi)
                            
                        # Add to the report
                        for key in ('id','slide','label'):
                            patch_data[key].append(roi_dict[key])
                        patch_data['contrast'].append(k)
                        patch_data['scale'].append(scale)
                        patch_data['radius'].append(1.0 / scale if scale > 0 else float('inf'))
                        patch_data['local_max'].append(d_local_max)
                        
        # Write the per-patch data
        df = pd.DataFrame(data=patch_data)
        df.to_csv(self._fn_sampling_roi_local_max_stats(work_dir, slide_id))
    
    def run_on_slide(self,
                     slide: Slide,
                     output_dir:str,
                     newer_only:bool=False):
            """Run WildCat on sampling ROIs from one slide.
            
            Args:
                slide (phas.client.api.Slide): A slide on PHAS for which sampling ROIs
                    are defined and for which inference will be performed. The ``task`` attribute
                    of the slide must be of type ``phas.client.api.SamplingROITask``.
                output_dir (str): Path to the output folder where heat maps will be stored
                newer_only (bool, optional): Use header information to avoid overwriting previous inference results
            """
        
            # Get the list of sampling ROIs for the subject we want to analyze
            rois = slide.task.slide_sampling_rois(slide.slide_id)

            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if the density already exists and is up to date
            curr_state = { 'rois': rois, 'dimensions': slide.dimensions, 'spacing': slide.spacing, 'slide_fn': slide.fullpath }
            state_file = self._fn_state(output_dir, slide.slide_id)
            skip_wildcat = False
            if newer_only:
                print(f'checking {state_file}')
                if os.path.exists(state_file):
                    with open(state_file) as state_fd:
                        stored_state = json.load(state_fd)
                        if stored_state == curr_state:
                            print(f'Skipping slide {slide.slide_id}')
                            skip_wildcat = True

            if not skip_wildcat:
                # Check if the slide filename is a gs:// URL, in which case the file has to be
                # downloaded using GCP
                if slide.fullpath.startswith('gs://'):
                    if self.gcs_client is None:
                        self.gcs_client = storage.Client()
                        self.gcs_cache = SelfManagedMultiFilePageCache()
                    osl = GoogleCloudOpenSlideWrapper(self.gcs_client, slide.fullpath, self.gcs_cache)
                else:
                    slide_fn_local = slide.fullpath
                    osl = openslide.OpenSlide(slide_fn_local)
                    
                # Create a datasource for the wildcat loader
                wc_datasource = OpenSlideHistologyDataSource(osl)

                # Get the unique labels in this slide
                unique_labels = set([x['label'] for x in rois])

                # Iterate over labels
                for label in unique_labels:
                    for roi_index, roi in enumerate([x for x in rois if x['label'] == label]):
                        geom_data = json.loads(roi['json'])

                        # Compute the bounding box of the trapezoid or polygon
                        (x_min, y_min, x_max, y_max) = (int(x + .5) for x in compute_sampling_roi_bounding_box(geom_data))
                        
                        # Apply padding to the bounding box
                        raw_spacing = slide.spacing
                        padpx = int(np.ceil(self.param.roi_padding * self.param.target_resolution / raw_spacing[0]))
                        (x_min, y_min, x_max, y_max) = (x_min - padpx, y_min - padpx, x_max + padpx, y_max + padpx)
                        
                        # Crop the bounding box by the slide dimensions
                        crop_result = crop_region_by_extents((x_min, y_min, x_max, y_max), (0, 0, slide.dimensions[0], slide.dimensions[1]))
                        if not crop_result:
                            print(f'Sampling ROI {roi["id"]} with label {label} in slide {slide.slide_id} is outside of slide!!!')
                            continue
                        (x_min, y_min, x_max, y_max) = crop_result

                        # For wildcat, region must be specified in units of windows or in relative units.
                        x_min_r, x_max_r = x_min / slide.dimensions[0], x_max / slide.dimensions[0]
                        y_min_r, y_max_r = y_min / slide.dimensions[1], y_max / slide.dimensions[1]
                        region_r = [x_min_r, y_min_r, x_max_r-x_min_r, y_max_r-y_min_r]
                        print(f'Slide {slide.slide_id}: sampling image region {(x_min_r,y_min_r)} to {(x_max_r,y_max_r)}')
                        
                        # Apply Wildcat on this region
                        t0 = time.time()
                        dens_wrgb = self.wildcat.apply(
                            osl=wc_datasource, 
                            window_size=self.param.window, 
                            extra_shrinkage=self.param.downsample, 
                            region=region_r, 
                            target_resolution=self.param.target_resolution,
                            crop=True, add_rgb_to_density=True)
                        t1 = time.time()
                        
                        # Split into density and non-density components
                        n_dens_comp = dens_wrgb.GetNumberOfComponentsPerPixel()-3
                        dens = sitk.GetImageFromArray(sitk.GetArrayFromImage(dens_wrgb)[:,:,:n_dens_comp], True)
                        dens.CopyInformation(dens_wrgb)
                        rgb = sitk.GetImageFromArray(sitk.GetArrayFromImage(dens_wrgb)[:,:,n_dens_comp:], True)
                        rgb.CopyInformation(dens_wrgb)
                        
                        # Create a blank image to store segmentation of the trapezoid
                        d_size, d_spacing, d_origin = dens.GetSize(), dens.GetSpacing(), dens.GetOrigin()
                        seg = Image.new('L', (d_size[0], d_size[1]))
                        
                        # Compute an affine transformation that maps the coordinates of the sampling ROI,
                        # which are in raw pixel units to the pixel units of the density image
                        M = np.zeros((2,3))
                        for i in range(2):
                            M[i,i] = raw_spacing[i] / d_spacing[i]
                            M[i,2] = (0.5 * raw_spacing[i] - d_origin[i]) / d_spacing[i]
                        geom_data = affine_transform_roi(geom_data, M)

                        # Spacing scaling factors used for drawing trapezoid
                        # sf = [raw_spacing[i]/d_spacing[i] for i in range(2)]
                        draw_sampling_roi(seg, geom_data, 1, 1, fill=1)
                        t2 = time.time()

                        # Create a sitk image to store the segmentation
                        seg_itk = sitk.GetImageFromArray(np.array(seg, dtype=np.uint8))
                        seg_itk.CopyInformation(dens)
                        
                        # Write the density image and the segmentation
                        sitk.WriteImage(dens, self._fn_density(output_dir, slide.slide_id, roi))
                        sitk.WriteImage(seg_itk, self._fn_mask(output_dir, slide.slide_id, roi))
                        sitk.WriteImage(rgb, self._fn_rgb(output_dir, slide.slide_id, roi))
                        t3 = time.time()

                        print(f'Completed in {t3-t0:6.4f}s, wildcat: {t1-t0:6.4f}s, draw: {t2-t1:6.4f}s, write: {t3-t2:6.4f}s')
                
                # Store the state for the future
                with open(state_file, 'wt') as state_fd:
                    json.dump(curr_state, state_fd)
                    
            # Compute the ROI scores on the output folder
            self.compute_superpixel_stats(output_dir, slide.slide_id)
            # self.compute_multiscale_mass_stats(output_dir, slide.slide_id)
        
    def run_on_task(self,
                    task: SamplingROITask,
                    output_dir: str,
                    specimens = None,
                    stains = None,
                    newer_only:bool=False):
        """Extract sampling ROIs from a task.
        
        Args:
            task (phas.client.api.SamplingROITask): Task on PHAS in which sampling ROIs
                are defined and for which inference will be performed
            output_dir (str): Path to the output folder where heat maps will be stored
            specimens (list of str, optional): Limit inferece to a set of specimens
            stains (list of str, optional): Limit inferece to a set of stains
            newer_only (bool, optional): Use header information to avoid overwriting previous inference results
        """
        
        # Start by getting a listing of slides for this task
        manifest = pd.DataFrame(task.slide_manifest(min_sroi=1)).set_index('id')
        
        # Filter out specimens and stains that we do not want
        if specimens:
            manifest = manifest.loc[manifest.specimen_private.isin(specimens)]
        if stains:
            manifest = manifest.loc[manifest.stain.isin(stains)]
        
        # Iterate over slides in the file
        for slide_id, _ in manifest.iterrows():
            slide = Slide(task, slide_id)
            slide_dir = self._wdir(output_dir, slide)
            self.run_on_slide(slide, slide_dir, newer_only=newer_only)
            
            
def wildcat_inference_on_sampling_rois_main(args):
    
    conn = Client(args.server, args.apikey, verify=not args.no_verify)
    task = SamplingROITask(conn, args.task)
    wildcat = TrainedWildcat(args.model)
    param = WildcatInferenceParameters(**yaml.safe_load(args.param))
    extractor = WildcatInferenceOnSamplingROI(wildcat, param)    

    extractor.run_on_task(task=task, output_dir=args.outdir, 
                            specimens=args.specimens, stains=args.stains, 
                            newer_only=args.newer)