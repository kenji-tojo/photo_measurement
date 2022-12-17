import rawpy
import imageio
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

UINT16_MAX = np.iinfo(np.uint16).max

def linear_rgb_to_linear_y(image_rgb: np.ndarray) -> np.ndarray:
    assert len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3
    return .2126 * image_rgb[:,:,0] + .7152 * image_rgb[:,:,1] + .0722 * image_rgb[:,:,2]

def convert_file(in_path: str) -> str:
    raw = rawpy.imread(in_path)
    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)

    out_dir = os.path.join(os.path.dirname(in_path), 'tiff')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.basename(in_path).split('.')[0] + '.tiff'
    out_path = os.path.join(out_dir, out_file)
    imageio.imsave(out_path, rgb)
    return out_path


def convert_dir(in_dir: str) -> None:
    if not os.path.exists(in_dir):
        print('directory does not exists')
        return False
    
    for raw_image_name in os.listdir(in_dir):
        if len(raw_image_name.split('.')) != 2 or raw_image_name.split('.')[1] != 'cr2':
            continue

        out_path = convert_file(os.path.join(in_dir, raw_image_name))
        print(f'saved to {out_path}')

        img = np.array(cv2.imread(out_path, cv2.IMREAD_UNCHANGED))
        img = np.array(img, dtype=np.float32) / UINT16_MAX
        img = linear_rgb_to_linear_y(img)
        print(f'max: {np.max(img.flatten())}')
        out_path = os.path.join(os.path.dirname(out_path), raw_image_name.split('.')[0] + '_gray.tiff')
        imageio.imsave(out_path, (img * UINT16_MAX).astype(np.uint16))

        img = img.flatten()
        size = len(img)
        hist, _ = np.histogram(img, bins=100)
        img = np.sort(img)
        print(f'remove {hist[0]} / {len(img)} pixels as lower outliers')
        img = img[hist[0]:]
        upper = np.quantile(img, .99)
        print(f'remove {len(img[img >= upper])} / {len(img)} pixels as upper outliers')
        img = img[img < upper]
        plt.clf()
        plt.hist(img, bins=100)
        out_path = os.path.join(os.path.dirname(out_path), raw_image_name.split('.')[0] + '_gray_hist.png')
        plt.savefig(out_path)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    args = parser.parse_args()
    convert_dir(args.input_directory)
