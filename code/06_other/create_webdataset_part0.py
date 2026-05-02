'''
Download Aesthetics 6.5+ Webdataset
'''
import os
import subprocess

base_url = "https://huggingface.co/akameswa/improved_aesthetics_6.5plus_webdataset/resolve/main"

for n in range(64):
    file = f'{str(n).zfill(5)}.tar'
    url  = os.path.join(base_url, file)
    print(url)
    subprocess.run(["wget", url])
    subprocess.run(["mv", file, "/home/datasets/laion_aesthetics_65plus/."])