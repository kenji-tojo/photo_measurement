import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

UINT16_MAX = np.iinfo(np.uint16).max

def linear_rgb_to_linear_y(image_rgb: np.ndarray) -> np.ndarray:
    assert len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3
    return .2126 * image_rgb[:,:,0] + .7152 * image_rgb[:,:,1] + .0722 * image_rgb[:,:,2]
    

def load_linear16tiff(in_path: str, print_info: bool = False):
    print(in_path)
    assert len(in_path) > 5 and in_path[-5:] == '.tiff'
    img = np.array(cv2.imread(in_path, cv2.IMREAD_UNCHANGED))
    if print_info:
        print(f'--- loaded image from {in_path} ---')
        print(f'dtype = {img.dtype}')
        [width, height, channels] = img.shape
        print(f'{width}x{height}x{channels}')
        img = img.astype(np.float64) / UINT16_MAX
        average = np.mean(img.flatten())
        print(f'average = {average}')
        print('------')
    return img

def load_from_directory(in_dir: str):
    if not os.path.exists(in_dir):
        print('directory does not exists')
        assert False
    for image_name in os.listdir(in_dir):
        if len(image_name.split('.')) != 2 or image_name.split('.')[1] != 'tiff':
            continue
        in_path = os.path.join(in_dir, image_name)
        load_linear16tiff(in_path, print_info=True)

def load_and_mask(in_path: str, thresh: float = 5e3):
    if not (len(in_path) > 5 and in_path[-5:] == '.tiff'):
        assert False
    
    img = load_linear16tiff(in_path, print_info=True)
    assert len(img.shape) == 3
    img_gray = linear_rgb_to_linear_y(img)
    print(f'max: {np.max(img_gray.flatten())}')
    img = np.repeat(img_gray[:,:,None], 3, axis=2)
    mask = (img_gray > (thresh / UINT16_MAX)).astype(np.float64)
    inv_mask = 1. - mask
    mask = np.repeat(mask[:,:,None], 3, axis=2)
    inv_mask = np.repeat(inv_mask[:,:,None], 3, axis=2)
    red = np.repeat(np.array([[0,.8,0]]), img_gray.shape[0], axis=0)
    red = np.repeat(red[:,None,:], img_gray.shape[1], axis=1)
    img = img * mask + inv_mask * red

    plt.clf()
    plt.imshow(img.astype(np.float32))
    plt.show()
    
    return True
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='load images from a directory')
    parser.add_argument('-f', '--file', help='load an image from a file')
    args = parser.parse_args()
    if args.directory is not None:
        load_from_directory(args.directory)
    elif args.file is not None:
        load_and_mask(args.file)
    else:
        assert False

