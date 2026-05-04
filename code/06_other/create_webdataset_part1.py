'''
Create a CSV file with all pairs (image file name, caption) 
of the LAION Aesthetics 6.5+ Webdataset.

TODO:
* Requires the LAION Aesthetics 6.5+ files (jpg, txt, json).
* Modify 'input_dir', 'output_dir'.
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

    path  = os.path.join(input_dir, f'*.{image_extension}')

    files = glob.glob(path)
    print(f'length files: {len(files)}')
    
    # Initialize separate lists for each column

    urls  = []
    texts = []

    for n, file in enumerate(files):
        base_name = file.split(".")[0]

        file_txt = f'{base_name}.txt'
        with open(file_txt, 'r') as fileTxt:
            caption = fileTxt.read()

        url = f'file://{file}'
        urls.append(url)
        texts.append(caption)

        if (n%100 == 0):
            print('.', end='')

        if (n%1000 == 0):
            print(f' {n}')


    # Create the DataFrame from the two lists
    df = pd.DataFrame({'URL': urls, 'TEXT': texts})


    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
