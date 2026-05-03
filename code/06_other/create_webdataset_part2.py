'''
Create tar files corresponding to Webdataset format containing 
a selected number (8192) of pairs (image file name, caption) from 
the Aesthetics 6.5+ dataset.
'''
from   img2dataset import download
import shutil
import os
import glob
import pandas as pd

if __name__ == "__main__":

    input_dir       = "OUR_DATASETS_DIR/laion_aesthetics_65plus/"
    output_dir      = "OUR_DATASETS_DIR/laion_aesthetics_65p_wds/"
    csv_file        = "laion_aesthetics_65plus_list_files.csv"
    image_extension = "jpg"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        processes_count         = 8,
        thread_count            = 16,
        url_list                = csv_file,
        image_size              = 256,
        output_folder           = output_dir,
        output_format           = "webdataset",
        input_format            = "csv",
        url_col                 = "URL",
        caption_col             = "TEXT",
        enable_wandb            = True,
        number_sample_per_shard = 8192,
        distributor             = "multiprocessing",
    )


    '''
    # Equivalent to the following shell command: 
    img2dataset \
        --url_list=laion_aesthetics_65plus_list_files.csv \
        --output_folder=OUR_DATASETS_DIR/laion_aesthetics_65p_wds/ \
        --processes_count=8 \
        --thread_count=16 \
        --image_size=256 \
        --input_format=csv \
        --output_format=webdataset \
        --url_col=uURL \
        --caption_col=TEXT \
        --enable_wandb=True \
        --number_sample_per_shard=8192 \
        --distributor=multiprocessing
    '''