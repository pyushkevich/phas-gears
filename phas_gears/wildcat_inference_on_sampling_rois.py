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

class WildcatInferenceOnSamplingROI:

    def __init__(self, wildcat:TrainedWildcat, suffix):
        self.wildcat = wildcat
        self.gcs_client = None
        self.gcs_cache = None
        self.suffix = suffix
    
    def run_on_slide(self,
                     slide: Slide,
                     output_dir:str, 
                     target_resolution:float=0.0004,
                     roi_padding:int=0, 
                     window:int = 1024,
                     shrink:int = 8,
                     newer_only:bool=False):
            """Run WildCat on sampling ROIs from one slide.
            
            Args:
                slide (phas.client.api.Slide): A slide on PHAS for which sampling ROIs
                    are defined and for which inference will be performed. The ``task`` attribute
                    of the slide must be of type ``phas.client.api.SamplingROITask``.
                output_dir (str): Path to the output folder where heat maps will be stored
                target_resolution (float, optional): Resolution in mm to which slides should be 
                    resampled before running Wildcat. Defaults to 0.0004 (0.4µm)
                roi_padding (int, optional): Additional padding around the ROI, in pixels. Defaults to 0.
                window (int, optional): Window size when performing inference, defaults to 1024
                shrink (int, optional): Shrink factor for output density map, defaults to 8
                newer_only (bool, optional): Use header information to avoid overwriting previous inference results
            """
        
            # Get the list of sampling ROIs for the subject we want to analyze
            rois = slide.task.slide_sampling_rois(slide.slide_id)

            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if the density already exists and is up to date
            curr_state = { 'rois': rois, 'dimensions': slide.dimensions, 'spacing': slide.spacing, 'slide_fn': slide.fullpath }
            state_file = os.path.join(output_dir, 'sroi_state.json')
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
                    padpx = int(np.ceil(roi_padding * target_resolution / raw_spacing[0]))
                    (x_min, y_min, x_max, y_max) = (x_min - padpx, y_min - padpx, x_max + padpx, y_max + padpx)

                    # Define the base filename for the output
                    fn_base = os.path.join(output_dir, f'{slide.slide_id}_label_{label:03d}_patch_{i:03d}')

                    # For wildcat, region must be specified in units of windows or in relative units.
                    x_min_r, x_max_r = max(0.0, x_min / slide.dimensions[0]), min(1.0, x_max / slide.dimensions[0])
                    y_min_r, y_max_r = max(0.0, y_min / slide.dimensions[1]), min(1.0, y_max / slide.dimensions[1])
                    region_r = [x_min_r, y_min_r, x_max_r-x_min_r, y_max_r-y_min_r]
                    print(f'Slide {slide.slide_id}: sampling image region {(x_min_r,y_min_r)} to {(x_max_r,y_max_r)}')
                    
                    # Apply Wildcat on this region
                    t0 = time.time()
                    dens = self.wildcat.apply(osl=wc_datasource, 
                                              window_size=window, 
                                              extra_shrinkage=shrink, 
                                              region=region_r, 
                                              target_resolution=target_resolution,
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
                    dname = 'density' + f'{self.suffix}' if self.suffix is not None else ''
                    sitk.WriteImage(dens, f'{fn_base}_{dname}.nii.gz')
                    sitk.WriteImage(seg_itk, f'{fn_base}_mask.nii.gz')
                    t3 = time.time()

                    print(f'Completed in {t3-t0:6.4f}s, wildcat: {t1-t0:6.4f}s, draw: {t2-t1:6.4f}s, write: {t3-t2:6.4f}s')
        
    def run_on_task(self,
                    task: SamplingROITask,
                    output_dir: str,
                    specimens = None,
                    stains = None,
                    **kwargs):
        """Extract sampling ROIs from a task.
        
        Args:
            task (phas.client.api.SamplingROITask): Task on PHAS in which sampling ROIs
                are defined and for which inference will be performed
            output_dir (str): Path to the output folder where heat maps will be stored
            specimens (list of str, optional): Limit inferece to a set of specimens
            stains (list of str, optional): Limit inferece to a set of stains
            **kwargs: See ``run_on_slide``.
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
            self.run_on_slide(slide, slide_dir, **kwargs)

    
if __name__ == '__main__':

    # Argument parser section
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', type=str, help='PHAS server URL', required=True)
    parser.add_argument('-k', '--apikey', type=str, help='JSON file with PHAS API key (also can set via PHAS_AUTH_KEY)')
    parser.add_argument('-v', '--no-verify', type='store_true', help='Disable SSL key verification')
    parser.add_argument('-t', '--task', type=int, help='PHAS task id', required=True)
    parser.add_argument('-m', '--model', type=str, help='Path to the Wildcat trained model')
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output folder', required=True)
    parser.add_argument('-z', '--scale', type=float, help='target resolution of the patch, mm/pixel', default=0.0004)
    parser.add_argument('-P', '--padding', type=int, help='extra padding for each ROI, pixels', default=0)
    parser.add_argument('-w', '--window-size', type=int, help='Window size for Wildcat inference, pixels', default=None)
    parser.add_argument('-w', '--shrink', type=int, help='Shrink factor for Wildcat inference', default=None)
    parser.add_argument('-n', '--newer', action="store_true", help='No overriding of existing results unless ROIs have changed')
    args = parser.parse_args()

    conn = Client(args.server, args.apikey)
    task = SamplingROITask(conn, args.task)
    wildcat = TrainedWildcat(args.model)
    extractor = WildcatInferenceOnSamplingROI(wildcat)
    extractor.run_on_task(task=task, output_dir=args.outdir, specimens=args.specimens, stains=args.stains, 
                          target_resolution=args.scale, roi_padding=args.padding, window=args.window_size,
                          shrink=args.shrink, newer_only=args.newer)