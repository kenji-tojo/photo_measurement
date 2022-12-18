import imageio
import os
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

def remap_color(path: str) -> None:
    if not os.path.exists(path):
        assert False
    
    image = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    image = image.astype(np.float32)
    image = image / np.iinfo(np.uint16).max

    dirname = os.path.dirname(path)
    imname = os.path.basename(path).split('.')[0]
   
    plt.clf()
    plt.imshow(image, cmap=mpl.colormaps['jet'])
    plt.savefig(os.path.join(dirname, imname + '_mapped.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to input file')
    args = parser.parse_args()
    remap_color(args.path)