# split_images.py
# Description: Transform 16-bit microscopy images to 8-bit and split them into patches.
# Author: Joshua Stiller
# Date: 16.10.24

from pathlib import Path

import cv2
import numpy as np
import polars as pl
import tifffile as tf


def preprocess_images(csv_path, output, patch_size=512):
    # Read the CSV file
    df = pl.read_csv(csv_path)

    for row in df.iter_rows(named=True):
        input_folder = Path(row['input_path'])
        channels = list(map(int, row['channels'].split(';')))
        channels = [c - 1 for c in channels]
        z_start = int(row['z_start'])
        z_end = int(row['z_end'])

        output_folder = Path(output)
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_path in input_folder.glob("*.tif"):
            image = tf.imread(str(image_path)).transpose(1, 0, 2, 3)
            for channel in channels:
                for z in range(z_start, z_end + 1):
                    img_sub = image[channel, z]
                    img_8bit = (img_sub / 256).astype(np.uint8)

                    # Normalize the 8-bit image per channel
                    img_normalized = cv2.normalize(img_8bit, None, 0, 255, cv2.NORM_MINMAX)

                    # Split the image into patches
                    h, w = img_normalized.shape
                    for i in range(0, h, patch_size):
                        for j in range(0, w, patch_size):
                            patch = img_normalized[i:i + patch_size, j:j + patch_size]

                            # Save each patch
                            patch_filename = f"{image_path.stem}_c{channel}_z{z}_p{i}-{j}.png"
                            patch_path = output_folder / patch_filename
                            cv2.imwrite(str(patch_path), patch)


if __name__ == "__main__":
    preprocess_images(
        csv_path="/Users/joshuastiller/Code/spotDetector/data/meta_data.csv",
        output="/Users/joshuastiller/Code/spotDetector/data/train",
        patch_size=512
    )
