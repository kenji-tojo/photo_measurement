import rawpy
import imageio
import os
import numpy as np
import cv2

UINT16_MAX = np.iinfo(np.uint16).max

def linear_rgb_to_linear_y(image_rgb: np.ndarray) -> np.ndarray:
    assert len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3
    return .2126 * image_rgb[:,:,0] + .7152 * image_rgb[:,:,1] + .0722 * image_rgb[:,:,2]

def convert_dir(in_dir: str):
    if not os.path.exists(in_dir):
        print('directory does not exists')
        return False
    
    for raw_image_name in os.listdir(in_dir):
        if len(raw_image_name.split('.')) != 2 or raw_image_name.split('.')[1] != 'cr2':
            continue

        raw = rawpy.imread(os.path.join(in_dir, raw_image_name))
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)

        out_dir = os.path.join(in_dir, 'tiff')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        tiff_image_name = raw_image_name.split('.')[0] + '.tiff'
        out_path = os.path.join(out_dir, tiff_image_name)
        print(f'saving {out_path}')
        imageio.imsave(out_path, rgb)

        img = np.array(cv2.imread(out_path, cv2.IMREAD_UNCHANGED))
        img = np.array(img, dtype=np.float64) / UINT16_MAX
        img = np.repeat(linear_rgb_to_linear_y(img)[:,:,None], 3, axis=2)
        print(f'max: {np.max(img.flatten())}')
        img = (img * UINT16_MAX).astype(np.uint16)
        out_path = os.path.join(out_dir, raw_image_name.split('.')[0] + '_gray.tiff')
        imageio.imsave(out_path, img)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    args = parser.parse_args()
    convert_dir(args.input_directory)
