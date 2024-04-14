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
torch.backends.cudnn.benchmark = True

from utils import utils_image as util
from utils.utils_video import VideoDecoder, VideoEncoder


def main():
    n_channels = 3

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='path to the model')
    parser.add_argument('--input', type=str, default='input', help='path of inputs')
    parser.add_argument('--output', type=str, default='output', help='path of results')
    parser.add_argument('--depth', type=int, default=16, help='bit depth of outputs')
    parser.add_argument('--suffix', type=str, default=None, help='output filename suffix')
    parser.add_argument('--video', type=str, default=None, help='ffmpeg video codec. if chosen, output video instead of images', choices=['dnxhd', 'libx264', 'libx265', '...'])
    parser.add_argument('--vprofile', type=str, default='high444', help='video profile')
    parser.add_argument('--crf', type=int, default=11, help='video crf')
    parser.add_argument('--preset', type=str, default='slow', help='video preset')
    parser.add_argument('--pix_fmt', type=str, default='yuv444p10le', help='video pixel format')
    parser.add_argument('--fps', type=str, default='24000/1001', help='video framerate')
    parser.add_argument('--res', type=str, default='1440:1080', help='video resolution to scale output to')
    parser.add_argument('--presize', action='store_true', help='resize video before processing')

    args = parser.parse_args()

    if not args.model_path:
        parser.print_help()
        raise ValueError('Please specify model_path')

    model_path = args.model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
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
        if not args.video:
            print('Error: input video requires --video to be set')
            return
    elif os.path.isdir(L_path):
        L_paths = util.get_image_paths(L_path)
    else:
        L_paths = [L_path]

    if args.video and (not E_path or os.path.isdir(E_path)):
        print('Error: output path must be a single video file')
        return

    if not os.path.exists(E_path) and os.path.splitext(E_path)[1] == '':
        util.mkdir(E_path)
    if not args.video and not os.path.isdir(E_path) and os.path.isdir(L_path):
        E_path = os.path.dirname(E_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    torch.cuda.empty_cache()

    from models.network_scunet import SCUNet as net
    model = net(state=torch.load(model_path))
    model.eval()
    scale = model.scale

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device).half()
    
    input_shape = (1, 3, 540, 720)
    dummy_input = torch.randn(input_shape).to(device).half()

    torch.cuda.empty_cache()
    
    # warmup
    with torch.no_grad():
        _ = model(dummy_input)

    print('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('Params number: {}'.format(number_parameters))

    print('model_name:{}'.format(model_name))
    print(L_path)

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    if args.suffix:
        suffix = f"{scale}x_{args.suffix}"
    else:
        suffix = f"{model_name}" if f"{scale}x_" in model_name else f"{scale}x_{model_name}"

    if video_input:
        video_decoder = VideoDecoder(L_path, options={'r': '24000/1001' }) # 'filter:v': 'yadif', 
        img_count = len(video_decoder)
        video_decoder.start()
    else:
        img_count = len(L_paths)

    if args.video:
        if '/' in args.fps:
            fps = Fraction(*map(int, args.fps.split('/')))
        elif '.' in args.fps:
            fps = float(args.fps)
        else:
            fps = int(args.fps)

        codec_options = {
            'crf':  str(args.crf),
            'preset': args.preset,
            'profile': args.vprofile,
            'pix_fmt': args.pix_fmt,
        }
        video_encoder = VideoEncoder(
            E_path,
            int(args.res.split(':')[0]),
            int(args.res.split(':')[1]),
            fps=fps,
            codec=args.video,
            pix_fmt=args.pix_fmt,
            options=codec_options,
            input_depth=args.depth,
        )
        video_encoder.start()

    image_names = []
    total_time = 0
    try:
        idx = 0
        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            if video_input:
                img_L = video_decoder.get_frame()
            elif len(L_paths) == 0:
                img_L = None
            else:
                img_L = L_paths.pop(0)
                img_name, ext = os.path.splitext(os.path.basename(img_L))
                img_L = util.imread_uint(img_L, n_channels=n_channels)
                image_names += [img_name]
            
            if args.presize:
                img_L = cv2.resize(img_L, (int(args.res.split(':')[0])//scale, int(args.res.split(':')[1])//scale), interpolation=cv2.INTER_CUBIC)
            
            img_L_t = util.uint2tensor4(img_L)
            img_L_t = img_L_t.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            
            #rng_state = torch.get_rng_state()
            #torch.manual_seed(13)
            
            img_E = model(img_L_t.half())
            #img_E, _ = util.tiled_forward(model, img_L_t, overlap=256, scale=scale)
            
            img_E = util.tensor2uint(img_E, args.depth)
            #torch.set_rng_state(rng_state)

            # ------------------------------------
            # save results
            # ------------------------------------
            if args.video:
                img_E = cv2.resize(img_E, (int(args.res.split(':')[0]), int(args.res.split(':')[1])), interpolation=cv2.INTER_CUBIC)

            if args.video:
                video_encoder.add_frame(img_E)
            elif os.path.isdir(E_path):
                util.imsave(img_E, os.path.join(E_path, f'{image_names.pop(0)}_{suffix}.png'))
            else:
                util.imsave(img_E, E_path)

            end.record()
            torch.cuda.synchronize()

            idx += 1
            time_taken = start.elapsed_time(end)
            total_time += time_taken
            time_remaining = ((total_time / (idx)) * (img_count - (idx+1)))/1000

            print(f'{idx}/{img_count}   fps: {1000/time_taken:.2f}  frame time: {time_taken:2f}ms   time remaining: {math.trunc(time_remaining/3600)}h{math.trunc((time_remaining/60)%60)}m{math.trunc(time_remaining%60)}s ', end='\r')
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, ending gracefully")
    except Exception as e:
        print("\n" + str(e))
    else:
        print("\n")

    if args.video:
        video_encoder.stop()
        video_encoder.join()
        if idx > 0:
            print(f"Saved video to {E_path}")
    if video_input:
        video_decoder.stop()
        video_decoder.join()

    if idx > 0:
        print(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {total_time / idx:.2f}ms per image              ')

if __name__ == '__main__':

    main()
