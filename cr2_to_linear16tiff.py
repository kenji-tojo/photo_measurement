import rawpy
import imageio
import os

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
        print(f'saving {os.path.join(out_dir, tiff_image_name)}')
        imageio.imsave(os.path.join(out_dir, tiff_image_name), rgb)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    args = parser.parse_args()
    convert_dir(args.input_directory)
