import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
# from pdb import set_trace as stx
import random
from utils import dstools, augmentor
from utils.augmentor import AddNoise, AddFixNoise, CalibratedNoiseParam, flip_and_route
from utils.dstools import (
    show_bgr, show_ir, RGBIR_with_channels, RawImageWithChannels,
    OV4686BayerPatternV2_RGB, OV4686BayerPatternV2_RGBIR, OV4686BayerPatternV2_IR,
    get_gt_rgb_with_channels, getshow_bgr, getshow_ir
)

NOISE_PARAM_FILE = './utils/noise_k.pkl'
NOISE_PARAM = CalibratedNoiseParam(NOISE_PARAM_FILE)
add_noise_func = AddNoise(NOISE_PARAM).do
add_fix_noise = AddFixNoise(NOISE_PARAM).do
to_rbgir_channels = RGBIR_with_channels().do

class DataLoaderTest(Dataset):
    def __init__(self, data_dir, sigma = 4):
        super(DataLoaderTest, self).__init__()

        files = os.listdir(os.path.join(data_dir, 'RGB'))

        self.rgb_filenames = [os.path.join(data_dir, 'RGB', x)  for x in files if x.endswith('npy')]
        self.nir_filenames = [os.path.join(data_dir, 'NIR', x) for x in files if x.endswith('npy')]

        self.sizex = len(self.rgb_filenames)  # get the size of target

        self.rng = np.random.RandomState()
        
        self.sigma = sigma

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        rgb_path = self.rgb_filenames[index_]
        nir_path = self.nir_filenames[index_]

        rgb = np.load(rgb_path)
        nir = np.load(nir_path)

        # adjust the image brightness
        rgb = to_rbgir_channels(rgb)

        gmean = (
                rgb['G11'] + rgb['G21'] + \
                rgb['G12'] + rgb['G22'] + \
                rgb['G13'] + rgb['G23'] + \
                rgb['G14'] + rgb['G24']
            ).mean() / 8
        for i in rgb:
            rgb[i] = rgb[i] * 5 / gmean

        # GT BGR
        gt_rgb_channels = RawImageWithChannels(
            get_gt_rgb_with_channels(rgb),
            OV4686BayerPatternV2_RGB)
        gt_bgr = np.stack([img for img in gt_rgb_channels.values()], axis=2)

        # add noise
        noisy_bgr = add_fix_noise(rgb, self.sigma)
        
        noisy_bgr = RawImageWithChannels(
            get_gt_rgb_with_channels(noisy_bgr),
            OV4686BayerPatternV2_RGB)
        noisy_bgr = np.stack([img for img in noisy_bgr.values()], axis=2)

        # Raw to sRGB
        inp_bgr = show_bgr(noisy_bgr)
        gt_bgr = show_bgr(gt_bgr)
        nir = show_ir(nir)

        inp_bgr = inp_bgr.transpose(2, 0, 1)
        gt_bgr = gt_bgr.transpose(2, 0, 1)
        nir = nir.transpose(2, 0, 1)

        inp_bgr = inp_bgr.copy()
        gt_bgr = gt_bgr.copy()
        nir = nir.copy()

        inp_bgr, gt_bgr, nir = inp_bgr.astype(np.float32), gt_bgr.astype(np.float32), nir.astype(np.float32)


        return inp_bgr, nir, gt_bgr




