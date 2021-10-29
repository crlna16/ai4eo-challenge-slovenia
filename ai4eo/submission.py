#!/usr/bin/env python
# Built-in modules
import os
import json
import datetime as dt
from typing import Tuple, List
import argparse

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd

# Imports from eo-learn and sentinelhub-py
from sentinelhub import CRS, BBox, SHConfig, DataCollection

from eolearn.core import (FeatureType,
                          EOPatch, 
                          EOTask, 
                          LinearWorkflow, 
                          EOExecutor, 
                          LoadTask,
                          SaveTask)
from eolearn.io import GeoDBVectorImportTask, SentinelHubInputTask, ExportToTiff

from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.mask import AddValidDataMaskTask

# making predictions
import torch

# Visualisation utilities from utils.py
from utils import get_extent

# our own modules
import eotasks
import model
from srresnet_ai4eo import ConvolutionalBlock, SubPixelConvolutionalBlock, ResidualBlock, SRResNet

def main(args):

    # --------------------
    # INPUT CHECK
    # --------------------

    assert args.s2_length==500
    assert args.batch_size==1

    # --------------------
    # USEFUL DEFINITIONS
    # --------------------

    # band specification
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
    band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from starter notebook
    band_wavelength = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1610, 2190] # nm
    band_spatial_resolution = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20] # m

    # EOPatches for input
    eops_train = os.listdir(args.raw_data_dir)

    # --------------------
    # DEFINE THE WORKFLOW
    # --------------------

    # load eopatches from raw data
    load_task = LoadTask(path=args.raw_data_dir)

    # compute reflectances
    compute_reflectances = eotasks.ComputeReflectances((FeatureType.DATA, 'BANDS'))

    # add the features: NDVI, NDWI, NDBI 
    ndvi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDVI'),
                                     [band_names.index('B08'), band_names.index('B04')])
    ndwi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDWI'),
                                     [band_names.index('B03'), band_names.index('B08')])
    ndbi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDBI'),
                                     [band_names.index('B11'), band_names.index('B08')])

    # validity masks
    # Validate pixels using SentinelHub's cloud detection mask and region of acquisition
    add_sh_validmask = AddValidDataMaskTask(eotasks.SentinelHubValidData(), 'IS_VALID')

    # Count the number of valid observations per pixel using valid data mask
    add_valid_count = eotasks.AddValidCountTask('IS_VALID', 'VALID_COUNT')

    # Filter out cloudy scenes
    valid_data_predicate = eotasks.ValidDataFractionPredicate(args.cloud_threshold)
    filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

    # Filter out nans
    nan_data_predicate = eotasks.NanDataPredicate()
    nan_filter_task = SimpleFilterTask((FeatureType.DATA, 'NDVI'), nan_data_predicate)

    # Predict
    model_state = torch.load(args.trained_model)
    args.input_channels = args.n_time_frames*(len(args.bands)+len(args.indices))
    eomodel = SRResNet(args)
    eomodel.load_state_dict(model_state)
    predict_task = eotasks.PredictPatchTask(eomodel, (FeatureType.DATA, 'BANDS'), args)
        
    # EXPORT PREDICTION AS TIFF - copied from starter notebook
    # Export the predicted binary mask as tiff for submission
    # NOTE: make sure both 0s and 1s are correctly exported
    export_task = ExportToTiff(feature=(FeatureType.MASK_TIMELESS, 'PREDICTION'),
                          folder=args.target_dir, crs='epsg:32633', image_dtype=np.uint8, no_data_value=255)

    save_task = SaveTask(path=args.target_dir, overwrite_permission=True)

    # construct the graph
    workflow = LinearWorkflow(load_task,
                              compute_reflectances,
                              ndvi,
                              ndwi,
                              ndbi,
                              add_sh_validmask,
                              add_valid_count,
                              filter_task,
                              nan_filter_task,
                              predict_task,
                              #save_task,
                              export_task
                              )

    # --------------------
    # EXECUTE THE WORKFLOW
    # --------------------

    # Define additional parameters of the workflow
    execution_args = []

    # need to specify the files that should be loaded 
    # loop all available eopatches and add each to the argument list
    execution_args = []

    eops_test = sorted(os.listdir(f'{args.raw_data_dir}/{args.flag}/'))

    for eop_test in eops_test:
        eop_exec_args = {
                load_task:   {'eopatch_folder': f'{args.flag}/{eop_test}'},
                save_task:   {'eopatch_folder': f'highres_{eop_test}'},
                export_task: {'filename': f'{eop_test}.tif'}
                        }
                                
        execution_args.append(eop_exec_args)

    # Execute the workflow
    executor = EOExecutor(workflow, execution_args, save_logs=True)
    executor.run(workers=args.n_processes, multiprocess=True)

    executor.make_report()
    print('Report was saved to location: {}'.format(executor.get_report_filename()))

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(f'Execution failed EOPatches with IDs:\n{failed_ids}\n'
                           f'For more info check report at {executor.get_report_filename()}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-random-seed', action='store_true', default=True, help='fixed random seed numpy / torch')
    parser.add_argument('--trained-model', type=str, default='best_model.pt')
    parser.add_argument('--raw-data-dir', type=str, default='/work/shared_data/2021-ai4eo/eopatches/')
    parser.add_argument('--target-dir', type=str, default='/work/shared_data/2021-ai4eo/submission/')
    parser.add_argument('--n-processes', type=int, default=1, help='Processes for EOExecutor')
    parser.add_argument('--cloud-threshold', type=float, default=0.9, help='threshold for valid data cloud mask')
    #parser.add_argument('--n-time-frames', type=int, default=1, help='Number of time frames in EOPatches')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output files')
    parser.add_argument('--flag', type=str, default='test', choices=["train", "test"])
    # need to pass model specific arguments ehre
    parser.add_argument('--n-valid-patches', type=int, default=10, help='Number of EOPatches selected for validation')
    parser.add_argument('--n-time-frames', type=int, default=1, help='Number of time frames in EOPatches')
    parser.add_argument('--filters', type=int, default=8)
    parser.add_argument('--s2-length', type=int, default=500, help='do not change this')
    parser.add_argument('--batch-size', type=int, default=1, help='do not change this')
    parser.add_argument('--scaling_factor', type=int, default=4)
    parser.add_argument('--n-channels', type=int, default=64)
    parser.add_argument('--bands', type=str, nargs='*', default=[], help='Sentinel band names (--> starter notebook')
    parser.add_argument('--indices', type=str, nargs='*', default=['NDVI'], choices=["NDVI", "NDWI", "NDBI"])
    parser.add_argument('--large_kernel_size', type=int, default=9)
    parser.add_argument('--small_kernel_size', type=int, default=3)
    parser.add_argument('--n-blocks', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-epochs', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience, -1 for no early stopping')
    parser.add_argument('--min-true-fraction', type=float, default=0, help='minimum fraction of positive pixels in target')
    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***\n')

    main(args)
