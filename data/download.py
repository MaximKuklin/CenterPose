# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import requests
import subprocess
import os
import tqdm
import argparse
from multiprocessing.pool import ThreadPool

public_url = "https://storage.googleapis.com/objectron"
categories = [
# "bike",
# "book",
# "bottle",
# "camera",
# "cereal_box",
"chair",
# "cup",
# "laptop",
# "shoe"
]


def download_file(video_id):

    video_filename = public_url + "/videos/" + video_id + "/video.MOV"
    metadata_filename = public_url + "/videos/" + video_id + "/geometry.pbdata"
    annotation_filename = public_url + "/annotations/" + video_id + ".pbdata"

    # If the file has been downloaded, skip it
    if os.path.exists(f"data/{c}/{video_id.replace('/', '_')}.pbdata"):
        return

    # video.content contains the video file.
    video = requests.get(video_filename)
    metadata = requests.get(metadata_filename)
    annotation = requests.get(annotation_filename)
    file = open(f"data/{c}/{video_id.replace('/', '_')}.MOV", "wb")
    file.write(video.content)
    file = open(f"data/{c}/{video_id.replace('/', '_')}_geometry.pbdata", "wb")
    file.write(metadata.content)
    file = open(f"data/{c}/{video_id.replace('/', '_')}.pbdata", "wb")
    file.write(annotation.content)
    file.close()


if __name__ == "__main__":

    # User defined parameters
    parser = argparse.ArgumentParser()

  
    parser.add_argument(
        '--c',
        nargs='+',
        default=categories,
        help = "categories to be generated"
    )
    opt = parser.parse_args()

    if not os.path.isdir('data/'):
        subprocess.call(["mkdir",'data/'])
    for c in opt.c:
        print(c)
        if not os.path.isdir(f'data/{c}'):
            subprocess.call(["mkdir",f'data/{c}'])
        blob_path = public_url + f"/v1/index/{c}_annotations"
        video_ids = requests.get(blob_path).text
        video_ids = video_ids.split('\n')

        results = ThreadPool(16).imap_unordered(download_file, video_ids)
        for _ in tqdm.tqdm(results, total=len(video_ids)):
            pass