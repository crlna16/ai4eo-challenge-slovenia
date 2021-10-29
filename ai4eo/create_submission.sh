#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /swork/caroline/jobs/ai4eo/all_indices_random_rot/epoch_100_model.pt --n-channels 125 --n-blocks 49 --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12  --n-time-frames 8 --indices NDVI NDWI NDBI --cloud-threshold 0.999
tar -C $1 -zcvf $1.tar.gz . 
#tar -zcvf submission.tar.gz . 
