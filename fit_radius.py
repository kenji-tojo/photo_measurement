import imageio
import os
import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt

UINT16_MAX = np.iinfo(np.uint16).max

def plot_radial(in_path: str) -> None:
    if not os.path.exists(in_path):
        print('input file does not exist')
        return
    
    image = np.array(cv2.imread(in_path, cv2.IMREAD_UNCHANGED))
    assert len(image.shape) == 2
    W, H = image.shape
    assert W == H
    print(W, H)
    r = W / 2

    image = image.astype(np.float32) / UINT16_MAX
    n_bins = 100
    counts = np.zeros(n_bins)
    sums = np.zeros(n_bins)

    for w in range(W):
        for h in range(H):
            dist = np.sqrt((w-r) ** 2 + (h-r) ** 2)
            sin = dist / r
            if sin >= 1.:
                continue
            i_bin = int(n_bins * sin)
            counts[i_bin] += 1
            sums[i_bin] += image[w, h]
    
    averages = sums / counts
    np.save('output/phong3_data', averages)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to input image')
    args = parser.parse_args()
    plot_radial(args.input)