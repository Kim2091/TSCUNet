import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import time
from glob import glob
import math
import torch
from datetime import timedelta
from fractions import Fraction

from utils.utils_video import VideoDecoder, VideoEncoder
from utils import utils_image as util


def load_frames(input_dir, clip_size, height=None, width=None):
    """Load a sequence of frames from a directory"""
    frame_paths = sorted(glob(os.path.join(input_dir, "*.png")) + 
                         glob(os.path.join(input_dir, "*.jpg")) + 
                         glob(os.path.join(input_dir, "*.jpeg")))
    
    if len(frame_paths) < clip_size:
        raise ValueError(f"Not enough frames in directory. Need at least {clip_size}, found {len(frame_paths)}")
    
    # Use the first clip_size frames
    frames = []
    for i in range(clip_size):
        img = cv2.imread(frame_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if dimensions specified
        if height and width:
            img = cv2.resize(img, (width, height))
            
        frames.append(img)
    
    return frames, frame_paths[:clip_size]


def preprocess_frames(frames):
    """Preprocess frames for model input"""
    # Convert to numpy array and normalize to [0, 1]
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    
    # Transpose to [time, channels, height, width]
    frames_np = np.transpose(frames_np, (0, 3, 1, 2))
    
    # Add batch dimension
    frames_np = np.expand_dims(frames_np, axis=0)
    
    return frames_np


def postprocess_frame(frame):
    """Convert model output back to a displayable image"""
    # Clip values to [0, 1]
    frame = np.clip(frame, 0, 1)
    
    # Convert to uint8
    frame = (frame * 255).astype(np.uint8)
    
    # Transpose from [channels, height, width] to [height, width, channels]
    frame = np.transpose(frame, (1, 2, 0))
    
    # Convert to BGR for OpenCV
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def run_inference(onnx_model_path, input_frames, verbose=True):
    """Run inference on a set of input frames using the ONNX model"""
    # Create ONNX Runtime session
    if verbose:
        print(f"Loading ONNX model from {onnx_model_path}")
    
    # Set session options
    sess_options = ort.SessionOptions()
    # Enable optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create inference session
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Get model metadata
    if verbose:
        print(f"Model inputs: {session.get_inputs()}")
        print(f"Model outputs: {session.get_outputs()}")
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    if verbose:
        print(f"Expected input shape: {input_shape}")
        print(f"Actual input shape: {input_frames.shape}")
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_frames})
    inference_time = time.time() - start_time
    
    if verbose:
        print(f"Inference completed in {inference_time:.4f} seconds")
        print(f"Output shape: {outputs[0].shape}")
    
    return outputs[0], inference_time


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser(description="Test TSCUNet ONNX model inference with video")
    parser.add_argument('--model_path', type=str, required=True, help='path to the ONNX model')
    parser.add_argument('--input', type=str, default='input', help='path of input video or directory')
    parser.add_argument('--output', type=str, default='output', help='path of output video or directory')
    parser.add_argument('--depth', type=int, default=8, help='bit depth of outputs')
    parser.add_argument('--suffix', type=str, default=None, help='output filename suffix')
    parser.add_argument('--video', type=str, default=None, help='ffmpeg video codec. if chosen, output video instead of images', 
                        choices=['dnxhd', 'h264_nvenc', 'libx264', 'libx265', '...'])
    parser.add_argument('--crf', type=int, default=11, help='video crf')
    parser.add_argument('--preset', type=str, default='slow', help='video preset')
    parser.add_argument('--fps', type=str, default=None, 
                        help='video framerate (defaults to input video\'s frame rate when processing video)')
    parser.add_argument('--res', type=str, default=None, help='video resolution to scale output to (optional, auto-calculated if not specified)')
    parser.add_argument('--presize', action='store_true', help='resize video before processing')
    parser.add_argument('--providers', type=str, default='CUDAExecutionProvider,CPUExecutionProvider', 
                        help='ONNX Runtime execution providers, comma separated')

    args = parser.parse_args()

    if not args.model_path:
        parser.print_help()
        raise ValueError('Please specify model_path')

    model_path = args.model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # ----------------------------------------
    # Input and output paths
    # ----------------------------------------
    L_path = args.input   # Input path
    E_path = args.output  # Output path

    if not L_path or not os.path.exists(L_path):
        print('Error: input path does not exist.')
        return
    
    # Check if input is a video file
    video_input = False
    if L_path.split('.')[-1].lower() in ['webm','mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mts', 
                                         'm2ts', 'ts', 'mov', 'qt', 'wmv', 'yuv', 'rm', 'rmvb', 'viv', 'asf', 'amv', 'mp4', 
                                         'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm2v', 'm4v', 'svi', '3gp', '3g2', 
                                         'mxf', 'roq', 'nsv', 'f4v', 'f4p', 'f4a', 'f4b']:
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
    
    # ----------------------------------------
    # Load ONNX model
    # ----------------------------------------
    print(f"Loading ONNX model from {model_path}")
    
    # Set up providers
    providers = [p.strip() for p in args.providers.split(',')]
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create inference session
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    # Get model metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Determine clip size from model input shape
    print(f"Model input shape: {input_shape}")
    
    # For temporal models, clip size is in the time dimension (dim 1 after batch)
    # Handle both static and dynamic shapes
    if len(input_shape) > 4:  # Ensure this is a temporal model (NTCHW format)
        clip_size = input_shape[1] if not isinstance(input_shape[1], str) and input_shape[1] > 0 else 5  # Default to 5 if dynamic
    else:
        raise ValueError("The provided model doesn't appear to be a temporal model (expected NTCHW format)")
    
    # Get required input dimensions, handling both fixed and dynamic shapes
    input_height_required = input_shape[3] if isinstance(input_shape[3], int) and input_shape[3] > 0 else None
    input_width_required = input_shape[4] if isinstance(input_shape[4], int) and input_shape[4] > 0 else None
    
    # Create test dimensions - use model's required dimensions if specified, otherwise use defaults
    test_height = input_height_required if input_height_required else 256
    test_width = input_width_required if input_width_required else 256
    
    print(f"Creating test input with shape (1, {clip_size}, 3, {test_height}, {test_width})")
    
    # Create test input with appropriate temporal shape
    test_input = np.zeros((1, clip_size, 3, test_height, test_width), dtype=np.float32)
    
    # Run test inference to determine scale factor
    test_output = session.run(None, {input_name: test_input})[0]
    print(f"Test output shape: {test_output.shape}")
    
    # Calculate scale factor based on output shape
    # For temporal models, compare output height to input height
    scale = test_output.shape[3] // test_height
    
    print(f"Model: {model_name}")
    print(f"Clip size: {clip_size}")
    print(f"Scale: {scale}x")
    print(f"Output shape from test: {test_output.shape}")

    # ----------------------------------------
    # Configure video decoder and resolution
    # ----------------------------------------
    n_channels = 3  # RGB
    
    if video_input:
        video_decoder = VideoDecoder(L_path, options={'r': '24000/1001'})
        img_count = len(video_decoder)
        video_decoder.start()
        
        # Get first frame to determine input resolution
        first_frame = video_decoder.get_frame()
        input_height, input_width = first_frame.shape[:2]
        # Reset video decoder
        video_decoder.stop()
        
        # Get input video's frame rate to use for output
        import av
        with av.open(L_path) as container:
            input_fps = container.streams.video[0].average_rate
        
        # Use the input video's frame rate for decoding
        video_decoder = VideoDecoder(L_path, options={'r': str(input_fps)})
        video_decoder.start()
    else:
        # For image input, get resolution from first image
        if len(L_paths) > 0:
            first_img = util.imread_uint(L_paths[0], n_channels=n_channels)
            input_height, input_width = first_img.shape[:2]
        else:
            print('Error: no input images found.')
            return

    # Calculate output resolution if not manually specified
    if args.res is None:
        if args.presize:
            # If presize is true, output resolution should match input resolution
            output_width = input_width
            output_height = input_height
        else:
            # Otherwise, scale up by model's scale factor
            output_width = input_width * scale
            output_height = input_height * scale
        output_res = f"{output_width}:{output_height}"
    else:
        output_res = args.res

    print(f"Input resolution: {input_width}x{input_height}")
    print(f"Output resolution: {output_res}")

    # ----------------------------------------
    # Process video/images
    # ----------------------------------------
    input_window = []
    image_names = []
    total_time = 0
    end_of_video = False
    video_encoder = None
    
    try:
        # Initialize video encoder if needed
        if args.video:
            if args.fps is None and video_input:
                # Use the input video's frame rate if not specified
                fps = input_fps
            elif args.fps is None:
                # Default for non-video inputs
                fps = Fraction(24000, 1001)
            elif '/' in args.fps:
                fps = Fraction(*map(int, args.fps.split('/')))
            elif '.' in args.fps:
                fps = float(args.fps)
            else:
                fps = int(args.fps)

            codec_options = {
                'crf': str(args.crf),
                'preset': args.preset,
            }
            video_encoder = VideoEncoder(
                E_path,
                int(output_res.split(':')[0]),
                int(output_res.split(':')[1]),
                fps=fps,
                codec=args.video,
                options=codec_options,
                input_depth=args.depth,
            )
            video_encoder.start()

        if args.suffix:
            suffix = f"{scale}x_{args.suffix}"
        else:
            suffix = f"{model_name}" if f"{scale}x_" in model_name else f"{scale}x_{model_name}"

        # Process frames
        idx = 0
        while True:
            import time
            start_time = time.time()
            
            # ------------------------------------
            # (1) Get input frame
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
            
            if img_L is None and not end_of_video:
                img_count = idx + clip_size // 2
                end_of_video = True
                # reflect pad the end of the window
                input_window += input_window[clip_size//2-1:-1][::-1]
            elif not end_of_video:
                if args.presize:
                    img_L = cv2.resize(img_L, (int(output_res.split(':')[0])//scale, int(output_res.split(':')[1])//scale), interpolation=cv2.INTER_CUBIC)
                
                # Convert to numpy array and normalize to [0, 1]
                img_L_np = img_L.astype(np.float32) / 255.0
                
                # Transpose to [channels, height, width]
                img_L_np = np.transpose(img_L_np, (2, 0, 1))

                input_window += [img_L_np]

            if len(input_window) < clip_size and end_of_video:
                # no more frames to process
                break
            elif len(input_window) < clip_size:
                # wait for more frames
                continue

            # ------------------------------------
            # (2) Run inference
            # ------------------------------------
            # Stack frames together for temporal model
            window_np = np.stack(input_window[:clip_size], axis=0)
            
            # Add batch dimension
            window_np = np.expand_dims(window_np, axis=0)
            
            # Check if we need to resize to match model's expected input dimensions
            curr_height, curr_width = window_np.shape[3], window_np.shape[4]
            
            # Get required dimensions (if specified in the model)
            required_height = input_height_required if input_height_required else curr_height
            required_width = input_width_required if input_width_required else curr_width
            
            # Resize if needed
            if curr_height != required_height or curr_width != required_width:
                print(f"Resizing input from {curr_height}x{curr_width} to {required_height}x{required_width}")
                
                # Create a new array with the required dimensions
                resized_window = np.zeros((1, clip_size, 3, required_height, required_width), dtype=np.float32)
                
                # For each frame in the clip
                for i in range(clip_size):
                    # Get the frame, reshape to HWC for OpenCV
                    frame = np.transpose(window_np[0, i], (1, 2, 0))  # CHW -> HWC
                    
                    # Resize with OpenCV
                    resized_frame = cv2.resize(frame, (required_width, required_height), 
                                             interpolation=cv2.INTER_CUBIC)
                    
                    # Convert back to CHW and store
                    resized_window[0, i] = np.transpose(resized_frame, (2, 0, 1))  # HWC -> CHW
                
                window_np = resized_window

            # Run ONNX inference
            outputs = session.run(None, {input_name: window_np.astype(np.float32)})
            
            # For temporal models, we typically use the center frame of the output sequence
            # Handle different output formats (with or without time dimension)
            if len(outputs[0].shape) > 4 and outputs[0].shape[1] > 1:
                # If output has time dimension with multiple frames, get center frame
                center_idx = outputs[0].shape[1] // 2
                img_E_np = outputs[0][0, center_idx]
            else:
                # Otherwise, just take the first frame
                img_E_np = outputs[0][0]  # Remove batch dimension
            
            # remove the oldest frame from the window for sliding window processing
            input_window.pop(0)

            # ------------------------------------
            # (3) Post-process output
            # ------------------------------------
            # Convert back to uint8/uint16 based on bit depth
            img_E_np = np.clip(img_E_np, 0, 1)
            if args.depth == 8:
                img_E = (img_E_np * 255).astype(np.uint8)
            else:
                img_E = (img_E_np * ((1 << args.depth) - 1)).astype(np.uint16)
            
            # Transpose from [channels, height, width] to [height, width, channels]
            img_E = np.transpose(img_E, (1, 2, 0))

            # ------------------------------------
            # (4) Save results
            # ------------------------------------
            if args.video:
                img_E = cv2.resize(img_E, (int(output_res.split(':')[0]), int(output_res.split(':')[1])), interpolation=cv2.INTER_CUBIC)
                video_encoder.add_frame(img_E)
            elif os.path.isdir(E_path):
                util.imsave(img_E, os.path.join(E_path, f'{image_names.pop(0)}_{suffix}.png'))
            else:
                util.imsave(img_E, E_path)

            # Calculate timing and progress
            end_time = time.time()
            time_taken = (end_time - start_time) * 1000  # Convert to ms
            total_time += time_taken
            
            idx += 1
            time_remaining = ((total_time / idx) * (img_count - idx)) / 1000

            print(f'{idx}/{img_count}   fps: {1000/time_taken:.2f}  frame time: {time_taken:.2f}ms   time remaining: {math.trunc(time_remaining/3600)}h{math.trunc((time_remaining/60)%60)}m{math.trunc(time_remaining%60)}s ', end='\r')
    
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, ending gracefully")
    except Exception as e:
        print("\n" + str(e))
    finally:
        # Clean up resources
        if video_encoder is not None:
            try:
                video_encoder.stop()
                # Add timeout to join to prevent hanging
                video_encoder.join(timeout=5)
                if idx > 0:
                    print(f"Saved video to {E_path}")
            except Exception as e:
                print(f"Error while closing video encoder: {e}")
            finally:
                # Force close the output container if still open
                if hasattr(video_encoder, 'output_container') and video_encoder.output_container:
                    try:
                        video_encoder.output_container.close()
                    except:
                        pass

        if video_input:
            try:
                video_decoder.stop()
                # Add timeout to join to prevent hanging
                video_decoder.join(timeout=5)
            except Exception as e:
                print(f"Error while closing video decoder: {e}")

        if idx > 0:
            print(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {total_time / idx:.2f}ms per image              ')

        # Force exit to ensure all threads are terminated
        os._exit(0)


if __name__ == '__main__':
    main()