from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml
import gdal
from osgeo import gdal, ogr

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='dped', type=str,
                    help='selecting different datasets')
parser.add_argument('--artifacts', default='', type=str,
                    help='selecting different artifacts type')
parser.add_argument('--cleanup_factor', default=2, type=int,
                    help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int,
                    choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('paths.yml', 'r') as stream:
    PATHS = yaml.full_load(stream)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def noise_patch(rgb_img, sp, max_var, min_mean):
    # convert to grayscale
    #img = rgb_img.convert('L')
    img = rgb2gray(rgb_img)

    # convert rgb and grayscale img to array
    rgb_img = np.array(rgb_img)
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i:i + sp, j:j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            if var_global < max_var and mean_global > min_mean:
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]
                collect_patchs.append(rgb_patch)

    return collect_patchs


if __name__ == '__main__':

    if opt.dataset == 'df2k':
        img_dir = PATHS[opt.dataset][opt.artifacts]['source']
        noise_dir = PATHS['datasets']['df2k'] + '/Corrupted_noise'
        sp = 256
        max_var = 20
        min_mean = 0
    # 1.
    else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = PATHS['datasets']['dped'] + '/noise_patches'
        # sp = 256 #size of noise patch / stride
        #max_var = 20
        #min_mean = 50

        # stride: change size of noise patch / stride to be less than (w/h - sp) so that for loops at line 38 and 39 can run at least 2 times
        # max_var and min_mean: values are chosen after analysing the var and mean of 25 patches (with and without NaN). This is near optimum for including good patches and exclude bad patches with lots of NaN
        sp = 128
        max_var = 0.0030
        min_mean = 0.08

    # create noise directory
    assert not os.path.exists(noise_dir)
    os.mkdir(noise_dir)

    # join sources images into a list
    #img_paths = sorted(glob.glob(osp.join(img_dir, '*.png')))
    img_paths = sorted(glob.glob(osp.join(img_dir, '*.tif')))
    cnt = 0

    for path in img_paths:
        img_name = osp.splitext(osp.basename(path))[0]
        print('**********', img_name, '**********')

        #img = Image.open(path).convert('RGB')
        im = gdal.Open(path, gdal.GA_ReadOnly)
        im_B2 = im.GetRasterBand(1).ReadAsArray()
        im_B3 = im.GetRasterBand(2).ReadAsArray()
        im_B4 = im.GetRasterBand(3).ReadAsArray()
        img = np.dstack([im_B2, im_B3, im_B4])

        patchs = noise_patch(img, sp, max_var, min_mean)
        for idx, patch in enumerate(patchs):
            save_path = osp.join(
                noise_dir, '{}_{:03}.png'.format(img_name, idx))
            cnt += 1
            print('collect:', cnt, save_path)
            #print('patch original: ', patch)
            patch_int = (patch * 255).astype(np.uint8)
            #print('patch_int:', patch_int)
            Image.fromarray(patch_int).save(save_path)
