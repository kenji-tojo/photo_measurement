import imageio
import os
import numpy as np
import cv2
from typing import Tuple

def clip_image(in_path: str, ratio: float) -> None:
    if not os.path.exists(in_path):
        print('input file does not exist')
        return
    
    image = np.array(cv2.imread(in_path, cv2.IMREAD_UNCHANGED))
    assert len(image.shape) >= 2
    width, height = image.shape[:2]
    half_width, half_height = .5 * width, .5 * height
    half_ratio = .5 * ratio
    w_start, w_end = int(half_width * (1. - half_ratio)), int(half_width * (1. + half_ratio))
    h_start, h_end = int(half_height * (1. - half_ratio)), int(half_height * (1. + half_ratio))
    image = image[w_start:w_end, h_start:h_end]

    dir = os.path.dirname(in_path)
    name = os.path.basename(in_path).split('.')[0]
    out_path = os.path.join(dir, name + '_clipped.tiff')
    imageio.imsave(out_path, image)

def clip_image_pix(in_path: str, range: Tuple[int]) -> None:
    if not os.path.exists(in_path):
        print('input file does not exist')
        return
    
    image = np.array(cv2.imread(in_path, cv2.IMREAD_UNCHANGED))
    image = image[range[2]:range[3], range[0]:range[1]]
    dir = os.path.dirname(in_path)
    name = os.path.basename(in_path).split('.')[0]
    out_path = os.path.join(dir, name + '_clipped.tiff')
    imageio.imsave(out_path, image)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to input image')
    parser.add_argument('-r', '--ratio', type=float, default=0.5, help='clip ratio')
    args = parser.parse_args()
    # clip_image(args.input, args.ratio)
    clip_image_pix(args.input, (666, 1170, 193, 697))
