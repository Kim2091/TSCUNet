from fractions import Fraction
import math
import os.path
import logging
import argparse
import av
import cv2

import numpy as np
from datetime import datetime, timedelta
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

def get_frame(container):
    try:
        for frame in container.decode(video=0):
            return frame.to_ndarray(format='rgb24')
    except:
        return None


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
    parser.add_argument('--video', type=str, default=None, help='video output codec. if not None, output video instead of images')
    parser.add_argument('--video_res', type=str, default='1440:1080', help='video resolution to scale output to')
    parser.add_argument('--video_presize', action='store_true', help='resize video to video_res/scale before processing')
    parser.add_argument('--avg_color_fix_scale', type=float, default=0.0, help='fix average color of output to match input')

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
    
    video_input = False
    if L_path.split('.')[-1].lower() in ['webm','mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mts', 'm2ts', 'ts', 'mov', 'qt', 'wmv', 'yuv', 'rm', 'rmvb', 'viv', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm2v', 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'f4v', 'f4p', 'f4a', 'f4b']:
        video_input = True
    elif os.path.isdir(L_path):
        L_paths = util.get_image_paths(L_path)
    else:
        L_paths = [L_path]

    if args.video and (not E_path or os.path.isdir(E_path)):
        print('Error: output path must be a single video file')
        return

    if not args.video and not os.path.isdir(E_path) and os.path.isdir(L_path):
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
    model = net(state=torch.load(model_path))
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

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    if args.suffix:
        suffix = f"{scale}x_{args.suffix}"
    else:
        suffix = f"{model_name}" if f"{scale}x_" in model_name else f"{scale}x_{model_name}"

    if video_input:
        input_container = av.open(L_path, options={'filter:v': 'yadif', 'r': '24000/1001'})
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = 'AUTO'
        # approximate number of frames with a couple seconds of headroom
        img_count = int((input_container.duration/1000000 + 10)*23.976)
    else:
        img_count = len(L_paths)


    if args.video:
        if args.video not in ('libx264', 'dnxhd'):
            print(f"Unsupported video codec: {args.video}")
            return

        options = {'c': args.video }

        if args.video == 'dnxhd':
            options['profile'] = 'dnxhr_444'
        elif args.video == 'libx264':
            options['profile'] = 'high444'
            options['crf'] = '13'
            options['preset'] = 'slow'
        else:
            print(f"Unsupported video codec: {args.video}")
            return
        
        output_container = av.open(E_path, mode='w')
        stream = output_container.add_stream(args.video, rate=Fraction(24000, 1001), options=options)
        stream.width = int(args.video_res.split(':')[0])
        stream.height = int(args.video_res.split(':')[1])
        stream.pix_fmt = "yuv444p10le"


    total_time = 0
    try:
        for idx in range(img_count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            if video_input:
                img_L = get_frame(input_container)
            elif len(L_paths) == 0:
                img_L = None
            else:
                img_L = L_paths.pop(0)
                img_name, ext = os.path.splitext(os.path.basename(img_L))
                img_L = util.imread_uint(img_L, n_channels=n_channels)
                
            if img_L is None:
                img_count = idx
                break

            if args.video and args.video_presize:
                img_L = cv2.resize(img_L, (int(args.video_res.split(':')[0])//scale, int(args.video_res.split(':')[1])//scale), interpolation=cv2.INTER_CUBIC)
            
            img_L_t = util.uint2tensor4(img_L)
            img_L_t = img_L_t.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            
            rng_state = torch.get_rng_state()
            torch.manual_seed(13)
            img_E, _ = util.tiled_forward(model, img_L_t, overlap=256, scale=scale)
            img_E = util.tensor2uint(img_E, args.depth)
            torch.set_rng_state(rng_state)

            # ------------------------------------
            # save results
            # ------------------------------------
            if args.video:
                img_E = cv2.resize(img_E, (int(args.video_res.split(':')[0]), int(args.video_res.split(':')[1])), interpolation=cv2.INTER_CUBIC)
            if args.avg_color_fix_scale > 0.01:
                img_E = util.avg_color_fix(img_E, img_L, args.avg_color_fix_scale)
            """
            if args.avg_color_fix_scale > 0.01:
                img_L_t = util.uint2tensor4(cv2.resize(img_L, ((int)(img_L.shape[1] * (args.avg_color_fix_scale / scale)), (int)(img_L.shape[0] * (args.avg_color_fix_scale / scale))), interpolation=cv2.INTER_CUBIC)).to(device)
                img_L_t, _ = util.tiled_forward(model, img_L_t, overlap=256, scale=scale)
                img_L = util.tensor2uint(img_L_t, args.depth)
                img_E = util.avg_color_fix(img_E, img_L, 1.0)
            """
            
            if args.video:
                for packet in stream.encode(av.VideoFrame.from_ndarray(img_E, format="rgb48le")):
                    output_container.mux(packet)
            elif os.path.isdir(E_path):
                util.imsave(img_E, os.path.join(E_path, f'{img_name}_{suffix}.png'))
            else:
                util.imsave(img_E, E_path)

            end.record()
            torch.cuda.synchronize()
            time_taken = start.elapsed_time(end)
            total_time += time_taken
            time_remaining = ((total_time / (idx+1)) * (img_count - (idx+1)))/1000

            print(f'{idx + 1}/{img_count}   fps: {1000/time_taken:.2f}  frame time: {time_taken:2f}ms   time remaining: {math.trunc(time_remaining/3600)}h{math.trunc((time_remaining/60)%60)}m{math.trunc(time_remaining%60)}s ', end='\r')
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, ending gracefully                    ")
    except av.error.EOFError:
        print("\nEnd of video reached                   ")

    if video_input:
        input_container.close()
    if args.video:
        for packet in stream.encode():
            output_container.mux(packet)
        output_container.close()
        print(f"Saved video to {E_path}             ")

    print(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {total_time / idx:.2f}ms per image              ')

if __name__ == '__main__':

    main()
