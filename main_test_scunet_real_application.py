import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
by Kai Zhang (2021/05-2021/11)
'''


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='scunet_color_real_psnr, scunet_color_real_gan')
    parser.add_argument('--model_path', type=str, default=None, help='path to the model')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_zoo', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--input', type=str, default='input', help='path of inputs')
    parser.add_argument('--output', type=str, default='output', help='path of results')
    parser.add_argument('--depth', type=int, default=8, help='bit depth of outputs')
    parser.add_argument('--suffix', type=str, default=None, help='output filename suffix')

    args = parser.parse_args()

    n_channels = 3

    if not args.model_name and not args.model_path:
        raise ValueError('Please specify either the model_name or model_path')

    if args.model_name == None:
        model_path = args.model_path
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    else:
        model_name = args.model_name
        model_path = os.path.join(args.model_zoo, model_name+'.pth')

    result_name = os.path.basename(args.input) + '_' + model_name     # fixed
    

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = args.input   # L_path, for Low-quality images
    E_path = args.output  # E_path, for Estimated images

    if not L_path or not os.path.exists(L_path):
        print('Error: input path does not exist.')
        return

    if not os.path.isdir(E_path) and os.path.isdir(L_path):
        E_path = os.path.dirname(E_path)
    if os.path.isdir(E_path) and not os.path.exists(E_path):
        util.mkdir(E_path)

    logger_name = result_name
    #utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_scunet import SCUNet as net
    model = net(state_dict=torch.load(model_path))
    model.eval()
    scale = model.scale

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    logger.info('model_name:{}'.format(model_name))
    logger.info(L_path)

    if os.path.isdir(L_path):
        L_paths = util.get_image_paths(L_path)
    else:
        L_paths = [L_path]

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    if args.suffix:
        suffix = f"{scale}x_{args.suffix}"
    else:
        suffix = f"{model_name}" if f"{scale}x_" in model_name else f"{scale}x_{model_name}"

    total_time = 0
    for idx, img in enumerate(L_paths):
        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))

        img_L = util.imread_uint(img, n_channels=n_channels)

        util.imshow(img_L) if args.show_img else None

        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        img_E, _ = util.tiled_forward(model, img_L, overlap=256, scale=scale)
        end.record()
        torch.cuda.synchronize()
        time_taken = start.elapsed_time(end)
        if idx > 0:
            total_time += time_taken

        img_E = util.tensor2uint(img_E, args.depth)

        # ------------------------------------
        # save results
        # ------------------------------------
        if os.path.isdir(E_path):
            util.imsave(img_E, os.path.join(E_path, f'{img_name}_{suffix}.png'))
        else:
            util.imsave(img_E, E_path)
        print(f'{idx + 1}/{len(L_paths)} in {time_taken:.2f}ms, time remaining: {(total_time / (idx+1)) * (len(L_paths) - (idx+1)):.2f}ms', end='\r')
    print(f'\nProcessed {len(L_paths)} images in {total_time:.2f}ms, average {total_time / len(L_paths):.2f}ms per image')

if __name__ == '__main__':

    main()
