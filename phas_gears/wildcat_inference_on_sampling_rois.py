from phas.client.api import Client, SamplingROITask, Slide
from phas.gcs_handler import GoogleCloudOpenSlideWrapper, SelfManagedMultiFilePageCache
from phas.dltrain import compute_sampling_roi_bounding_box, affine_transform_roi, draw_sampling_roi
from google.cloud import storage
from tangle_cnn.wildcat_main import TrainedWildcat
from tangle_cnn.osl_worker import OpenSlideHistologyDataSource
import openslide
import argparse
import pandas as pd
import numpy as np
import os
import json
import time
import SimpleITK as sitk
from PIL import Image   

import pymetis
from scipy.sparse import csr_matrix, eye
from sklearn.feature_extraction.image import grid_to_graph

import pydantic
import yaml

from pydantic import BaseModel
from typing import Dict, Literal, List

class WildcatInferenceParameters(BaseModel):
    """Parameters for WildCat inference"""
    
    #: Name of the stain
    stain: str
    
    #: Suffix for the output files, may include name of experiment, etc.
    suffix: str
    
    #: Contrasts to evaluate, in form 'name':class_id, where class_id is one of the classes in WildCat training
    contrasts: Dict [str, int]
    
    #: Resolution in mm to which slides should be resampled before running Wildcat.
    target_resolution = 0.0004
    
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


class WildcatInferenceOnSamplingROI:

    def __init__(self, wildcat:TrainedWildcat, param: WildcatInferenceParameters):
        self.wildcat = wildcat
        self.param = param
        self.gcs_client = None
        self.gcs_cache = None
        
    def _fn_state(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_state.json')

    def _fn_density(self, workdir, slide_id, label, patch):
        return os.path.join(workdir, f'slide_{slide_id}_label_{label:03d}_patch_{patch:03d}_density_{self.param.suffix}.nii.gz')
    
    def _fn_mask(self, workdir, slide_id, label, patch):
        return os.path.join(workdir, f'slide_{slide_id}_label_{label:03d}_patch_{patch:03d}_mask.nii.gz')
    
    def _fn_cluster_labels(self, workdir, slide_id, label, patch):
        return os.path.join(workdir, f'slide_{slide_id}_label_{label:03d}_patch_{patch:03d}_density_{self.param.suffix}_clusters.nii.gz')
        
    def _fn_cluster_heatmap(self, workdir, slide_id, label, patch, contrast):
        return os.path.join(workdir, f'slide_{slide_id}_label_{label:03d}_patch_{patch:03d}_density_{self.param.suffix}_clustermap_{contrast}.nii.gz')
    
    def _fn_sampling_roi_stats(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_stats_{self.param.suffix}.json')
        
    def _fn_sampling_roi_cluster_stats(self, workdir, slide_id):
        return os.path.join(workdir, f'slide_{slide_id}_sroi_cluster_activation_{self.param.suffix}.csv')
        
    def compute_superpixel_stats(self, work_dir, slide_id):
        
        # Look in the slide directory for the listing of ROIs
        with open(self._fn_state(slide_id)) as sroi_json_fd:
            rois = json.load(sroi_json_fd)['rois']

        # Get the list of unique labels
        unique_labels = set([x['label'] for x in rois])

        # Create a pandas frame to hold individual patch measures for validation
        patch_data = {'id':[], 'slide':[], 'label':[], 'patch':[], 'contrast':[], 'value':[]}

        # Repeat for each label
        quantiles = dict()
        for label in unique_labels:

            # List of tangle/thread values for all cells
            val = { k:[] for k in self.param.contrasts.keys() }
            n_cells = 0
            for i, roi_dict in enumerate([x for x in rois if x['label'] == label]):

                # Load the density image
                img_density = sitk.ReadImage(self._fn_density(work_dir, slide_id, label, i))

                # Load the mask image for this ROI
                img_roi = sitk.ReadImage(self._fn_mask(work_dir, slide_id, label, i))

                # We want the average area of a sampling region to be 200x200 microns
                pix_area = img_density.GetSpacing()[0] * img_density.GetSpacing()[1]
                pix_per_cluster = self.param.cluster_area / pix_area

                # Get the density maps for the tangles and threads (or whatever we are looking for)
                density = sitk.GetArrayFromImage(img_density)
                if self.param.normalization == 'minus_others':
                    d = { k: np.maximum(density[:,:,v] * 2 - density.sum(axis=2), 0) for k,v in self.param.contrasts.items() }
                else:
                    d = { k: np.maximum(density[:,:,v],0) for k,v in self.param.contrasts.items() }

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
                sitk.WriteImage(img_roi_part, self._fn_cluster_labels(work_dir, slide_id, label, i))

                # Compute the average tangles/threads in each ROI
                for k, d_roi_k in d.items():
                    # Create an image for this contrast where we assign each ROI the computed value
                    roi_hmap = np.zeros_like(roi, dtype=np.float32)
                    roi_hmap[roi==0] = np.nan

                    for j in range(np.max(roi_part)):
                        mask = np.where(roi_part==j+1, 1, 0)
                        d_roi_masked = np.maximum(d_roi_k * mask, 0)
                        f = np.sum(d_roi_masked) / np.sum(mask)
                        val[k].append(f)
                        roi_hmap[roi_part==j+1] = f
                        for key in ('id','slide','label'):
                            patch_data[key].append(roi_dict[key])
                        patch_data['patch'].append(j)
                        patch_data['contrast'].append(k)
                        patch_data['value'].append(f)
                        
                    # Save the ROI heat map
                    img_hmap = sitk.GetImageFromArray(roi_hmap, isVector=False)
                    img_hmap.CopyInformation(img_density)
                    sitk.WriteImage(img_hmap, self._fn_cluster_heatmap(work_dir, slide_id, label, i, k))

            # Compute the quantiles
            if n_cells > 0:
                quantiles[label] = { 'n_cells': n_cells }
                for k,v in val.items():
                    quantiles[label][k] = { q : np.quantile(v, q) for q in self.param.qtile_list }
                print(f'Label: {label}, Quantiles: {quantiles[label]}')

        # Write the summary statistics
        with open(self._fn_sampling_roi_stats(work_dir, slide_id), 'wt') as stats_fd:
            json.dump(quantiles, stats_fd)

        # Write the per-patch data
        df = pd.DataFrame(data=patch_data)
        df.to_csv(self._fn_sampling_roi_cluster_stats(work_dir, slide_id))
    
    
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
            state_file = self._fn_state(slide.slide_id)
            if newer_only:
                print(f'checking {state_file}')
                if os.path.exists(state_file):
                    with open(state_file) as state_fd:
                        stored_state = json.load(state_fd)
                        if stored_state == curr_state:
                            print(f'Skipping slide {slide.slide_id}')
                            return

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
                for i, roi in enumerate([x for x in rois if x['label'] == label]):
                    geom_data = json.loads(roi['json'])

                    # Compute the bounding box of the trapezoid or polygon
                    (x_min, y_min, x_max, y_max) = (int(x + .5) for x in compute_sampling_roi_bounding_box(geom_data))
                    
                    # Apply padding to the bounding box
                    raw_spacing = slide.spacing
                    padpx = int(np.ceil(self.param.roi_padding * self.param.target_resolution / raw_spacing[0]))
                    (x_min, y_min, x_max, y_max) = (x_min - padpx, y_min - padpx, x_max + padpx, y_max + padpx)

                    # For wildcat, region must be specified in units of windows or in relative units.
                    x_min_r, x_max_r = max(0.0, x_min / slide.dimensions[0]), min(1.0, x_max / slide.dimensions[0])
                    y_min_r, y_max_r = max(0.0, y_min / slide.dimensions[1]), min(1.0, y_max / slide.dimensions[1])
                    region_r = [x_min_r, y_min_r, x_max_r-x_min_r, y_max_r-y_min_r]
                    print(f'Slide {slide.slide_id}: sampling image region {(x_min_r,y_min_r)} to {(x_max_r,y_max_r)}')
                    
                    # Apply Wildcat on this region
                    t0 = time.time()
                    dens = self.wildcat.apply(osl=wc_datasource, 
                                              window_size=self.param.window, 
                                              extra_shrinkage=self.param.shrink, 
                                              region=region_r, 
                                              target_resolution=self.param.target_resolution,
                                              crop=True)
                    t1 = time.time()

                    '''
                    # img = osl.read_region((x_min, y_min), 0, (x_max-x_min,y_max-y_min))

                    # Resample the image to target resolution
                    w = int(0.5 + img.width * spc_x / target_resolution)
                    h = int(0.5 + img.height * spc_y / target_resolution)
                    spacing = img.width * spc_x / w, img.height * spc_y / h
                    print(f'Resampling to size {(w,h)} with spacing {spacing}')
                    img = img.resize((w, h), Image.BICUBIC)
                    t2 = time.time()
                    '''

                    # Create a blank image to store segmentation of the trapezoid
                    d_size, d_spacing, d_origin = dens.GetSize(), dens.GetSpacing(), dens.GetOrigin()
                    seg = Image.new('L', (d_size[0], d_size[1]))
                    
                    # Compute an affine transformation that maps the coordinates of the sampling ROI,
                    # which are in raw pixel units to the pixel units of the density image

                    # Shift the trapezoid to the xmin/ymin and draw trapezoid
                    # M = np.array([[1., 0., -x_min], [0., 1., -y_min]])
                    M = np.zeros((2,3))
                    for i in range(2):
                        M[i,i] = raw_spacing[i] / d_spacing[i]
                        M[i,2] = (0.5 * raw_spacing[i] - d_origin[i]) / d_spacing[i]
                    geom_data = affine_transform_roi(geom_data, M)
                    print(geom_data)

                    # Spacing scaling factors used for drawing trapezoid
                    # sf = [raw_spacing[i]/d_spacing[i] for i in range(2)]
                    draw_sampling_roi(seg, geom_data, 1, 1, fill=1)
                    t2 = time.time()

                    # Save the patch as a NIFTI file (should be optional)
                    '''
                    fn_base = os.path.join(slide_dir, f'{slide_str}_label_{label:03d}_patch_{i}')
                    os.makedirs(os.path.dirname(fn_base), exist_ok=True)
                    # img.convert('RGB').save(f'{fn_base}.tiff')
                    img_itk = sitk.GetImageFromArray(np.array(img.convert('RGB'), dtype=np.uint8), isVector=True)
                    img_itk.SetSpacing((spacing[0], spacing[1]))
                    sitk.WriteImage(img_itk, f'{fn_base}_rgb.nii.gz')
                    t4 = time.time()
                    '''

                    # Create a sitk image to store the segmentation
                    seg_itk = sitk.GetImageFromArray(np.array(seg, dtype=np.uint8))
                    seg_itk.CopyInformation(dens)
                    
                    # Write the density image and the segmentation
                    dname = 'density' + f'_{self.suffix}' if self.suffix is not None else ''
                    sitk.WriteImage(dens, self._fn_density(output_dir, slide.slide_id, label, i))
                    sitk.WriteImage(seg_itk, self._fn_mask(output_dir, slide.slide_id, label, i))
                    t3 = time.time()

                    print(f'Completed in {t3-t0:6.4f}s, wildcat: {t1-t0:6.4f}s, draw: {t2-t1:6.4f}s, write: {t3-t2:6.4f}s')
            
            # Store the state for the future
            with open(state_file, 'wt') as state_fd:
                json.dump(curr_state, state_fd)
                
            # Compute the ROI scores on the output folder
            self.compute_superpixel_stats(output_dir, slide.slide_id)
        
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
            slide_dir = os.path.join(output_dir, f'slide_{slide.slide_id}')
            self.run_on_slide(slide, slide_dir, newer_only=newer_only)

    
if __name__ == '__main__':

    # Argument parser section
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', type=str, help='PHAS server URL', required=True)
    parser.add_argument('-k', '--apikey', type=str, help='JSON file with PHAS API key (also can set via PHAS_AUTH_KEY)')
    parser.add_argument('-v', '--no-verify', type='store_true', help='Disable SSL key verification')
    parser.add_argument('-t', '--task', type=int, help='PHAS task id', required=True)
    parser.add_argument('-m', '--model', type=str, help='Path to the Wildcat trained model')
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output folder', required=True)
    parser.add_argument('-p', '--param', type=argparse.FileType('rt'), help='Parameter .yaml file')
    parser.add_argument('-n', '--newer', action="store_true", help='No overriding of existing results unless ROIs have changed')
    args = parser.parse_args()

    conn = Client(args.server, args.apikey, verify=~args.verify)
    task = SamplingROITask(conn, args.task)
    wildcat = TrainedWildcat(args.model)
    param = WildcatInferenceParameters(**yaml.safe_load(args.param))
    extractor = WildcatInferenceOnSamplingROI(wildcat, param)    
    
    extractor.run_on_task(task=task, output_dir=args.outdir, 
                          specimens=args.specimens, stains=args.stains, 
                          newer_only=args.newer)