import os
import numpy as np
import cv2

def load_linear16tiff(in_dir: str, print_info: bool = False):
    if not os.path.exists(in_dir):
        print('directory does not exists')
        assert False
    
    for image_name in os.listdir(in_dir):
        if len(image_name.split('.')) != 2 or image_name.split('.')[1] != 'tiff':
            continue

        in_path = os.path.join(in_dir, image_name)
        img = np.array(cv2.imread(in_path, cv2.IMREAD_UNCHANGED))
        if print_info:
            print(f'--- loaded image from {in_path} ---')
            print(f'dtype = {img.dtype}')
            [width, height, channels] = img.shape
            print(f'{width}x{height}x{channels}')
            img = img.astype(np.float64)
            average = np.mean(img.flatten())
            print(f'average = {average}')
            print('------')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    args = parser.parse_args()
    load_linear16tiff(args.input_directory, print_info=True)

