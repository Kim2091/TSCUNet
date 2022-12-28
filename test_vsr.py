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
from utils.utils_video import VideoEncoder, get_codec_options


'''
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
by Kai Zhang (2021/05-2021/11)
'''

_frames = {}
def get_frame(container):
    if hash(container) not in _frames:
        _frames[hash(container)] = []
    if len(_frames[hash(container)]) > 0:
        return _frames.pop(0)

    try:
        [_frames[hash(container)].append(frame.to_ndarray(format='rgb24')) for frame in container.decode(video=0)]
    except:
        return None

    return _frames[hash(container)].pop(0)

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
    parser.add_argument('--depth', type=int, default=16, help='bit depth of outputs')
    parser.add_argument('--suffix', type=str, default=None, help='output filename suffix')
    parser.add_argument('--video', type=str, default=None, help='video output codec. if not None, output video instead of images')
    parser.add_argument('--res', type=str, default='1440:1080', help='video resolution to scale output to')
    parser.add_argument('--presize', action='store_true', help='resize video to res/scale before processing')

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
    

    logger_name = result_name
    #utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_tscunet import TSCUNet as net
    model = net(state=torch.load(model_path))
    model.eval()
    scale = model.scale
    clip_size = model.clip_size

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device).half()

    torch.cuda.empty_cache()
    
    # warmup
    with torch.no_grad():
        _ = model(torch.randn(1, 5, 3, 128, 128).to(device).half())

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
        if args.video not in ('libx264', 'libx265', 'dnxhd'):
            print(f"Unsupported video codec: {args.video}")
            return

        pix_fmt, options = get_codec_options(args.video)
        video_encoder = VideoEncoder(
            E_path,
            int(args.res.split(':')[0]),
            int(args.res.split(':')[1]),
            fps=Fraction(24000, 1001),
            codec=args.video,
            pix_fmt=pix_fmt,
            options=options,
            input_depth=args.depth,
        )
        video_encoder.start()

    input_window = []
    image_names = []
    total_time = 0
    end_of_video = False
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
                img_L = get_frame(input_container)
            elif len(L_paths) == 0:
                img_L = None
            else:
                img_L = L_paths.pop(0)
                img_name, ext = os.path.splitext(os.path.basename(img_L))
                img_L = util.imread_uint(img_L, n_channels=n_channels)
                image_names += [img_name]
                
            if img_L is None and not end_of_video:
                img_count = idx + clip_size // 2
                end_of_video = True
                # reflect pad the end of the window
                input_window += input_window[clip_size//2-1:-1][::-1]
            elif not end_of_video:
                if args.presize:
                    img_L = cv2.resize(img_L, (int(args.res.split(':')[0])//scale, int(args.res.split(':')[1])//scale), interpolation=cv2.INTER_CUBIC)
            
                img_L_t = util.uint2tensor4(img_L)
                img_L_t = img_L_t.to(device)

                input_window += [img_L_t]

            if len(input_window) < clip_size and end_of_video:
                # no more frames to process
                break
            elif len(input_window) < clip_size // 2 + 1:
                # wait for more frames
                continue
            elif len(input_window) == clip_size // 2 + 1:
                # reflect pad the beginning of the window
                input_window = input_window[1:][::-1] + input_window

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            
            #rng_state = torch.get_rng_state()
            #torch.manual_seed(13)
            window = torch.stack(input_window[:clip_size], dim=1)
            
            tta = False
            if tta:
                h, w = window.shape[-2:]
                window = torch.cat([window, torch.flip(torch.roll(window, (h//2, w//2), (-2, -1)), [-4, -3, -2, -1])], dim=0) # tta

            img_E, _ = util.tiled_forward(model, window, overlap=256, scale=scale)
            del window

            if tta:
                # reverse tta
                img_E = torch.stack([img_E[0], torch.flip(torch.roll(img_E[1], (h, w), (-2, -1)), [-3, -2, -1])], dim=0)
                img_E = torch.mean(img_E, dim=0, keepdim=True)

            img_E = util.tensor2uint(img_E, args.depth)
            #torch.set_rng_state(rng_state)

            # remove the oldest frame from the window
            input_window.pop(0)

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
    except av.error.EOFError:
        print("\nEnd of video reached")
    else:
        print("\n")

    if video_input:
        input_container.close()
    if args.video:
        video_encoder.stop()
        video_encoder.join()
        print(f"Saved video to {E_path}")

    print(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {total_time / idx:.2f}ms per image              ')

if __name__ == '__main__':

    main()
