# TSCUNet - Temporal modification of the SCUNet Architecture

# Beginner's Guide

This is a simple guide to help you use TSCUNet for video upscaling.

## Getting Started

1. Clone or download this repository: 
   ```
   git clone https://github.com/Kim2091/SCUNet
   ```
   (Or download the zip file from GitHub)

2. Install PyTorch with CUDA from: https://pytorch.org/get-started/locally/

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Using the GUI

For easy use, you can launch the graphical interface:

```
python vsr_gui.py
```

### GUI Features

1. **Input Video**: Drop your video file or click to browse and select one.

2. **Model Selection**: Choose between PyTorch (.pth) or ONNX (.onnx) models and select the appropriate model file.

3. **Output**: Specify where to save the upscaled video.

4. **Options**:
   - **Presize**: Automatically resize the input to match your desired output ratio, which can improve quality.
   - **Video Codec**: Select from common codecs like h264_nvenc (NVIDIA GPU), libx264 (CPU), or specify a custom one.

5. **Processing**: Click "Process Video" to start upscaling. A progress bar will show the status, and you can monitor the details in the log area.

6. **Stop Processing**: If needed, you can stop an ongoing process.

That's it! The GUI provides a simple way to access all the functionality without needing to remember command-line options.

# Advanced Information
---------
To get started, simply: 
1. Clone this repository with `git clone https://github.com/Kim2091/SCUNet` (or download the zip). 
2. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
3. Run `pip install -r requirements.txt`
4. Use 

1. Video/Temporal Inference

    ```bash
    python test_vsr.py --model_path pretrained_models/2x_eula_anifilm_vsr.pth --input example/lr/ --output example/sr/ --depth 16
    ```

2. Single-Image inference

    ```bash
    python test_sisr.py --model_path pretrained_models/scunet_color_real_psnr.pth --input example/lr/ --output example/sr/ --depth 16
    ```
If a folder of images is provided as input, they all must match in resolution.
    
Both architectures support image inputs with video output and vice-versa. Input and output arguments can be a path to either a single image, a folder of images, or a video file. To output to a video, the `--video` argument must be provided to select the output video codec. Additional ffmpeg arguments such as `--profile`, `--preset`, `--crf`, and `--pix_fmt` can also be provided if desired. In the original repository by eula, the `--res` command was required. This is no longer necessary, as the script calculates the output resolution for you.

Additionally, the `--presize` argument can be used to resize the input to the target resolution divided by the scale, which can be produce better results when the output resolution is short of the target resolution or if the original aspect ratio does not match the target aspect ratio.
```bash
python test_vsr.py --model_path pretrained_models/tscu_2x.pth --input example/lr_video.mp4 --output example/sr_video.mp4 --video libx264 --presize
```

ONNX Model Conversion and Testing
----------
I added this on to hopefully allow for TensorRT or DirectML support in the near future. Having easy ONNX conversion should make implementing this easier.

1. Convert PyTorch model to ONNX format

    ```bash
    python convert_to_onnx.py --model pretrained_models/model.pth --output model.onnx --dynamic
    ```
    Optional arguments:
    - `--dynamic`: Outputs a dynamic onnx model, good for processing various sized input videos
    - `--height`, `--width`: Outputs a static onnx. Good for upscaling videos of specific resolutions. Specify input dimensions (e.g. 256)
    - `--batch`: Set batch size. Don't mess with this (default: 1)
    - `--no-optimize`: Disable optimization wrapper

2. Convert ONNX model to FP16 (provides a small speed boost)

    ```bash
    python onnx_fp32_to_fp16.py --model model.onnx --output model_fp16.onnx
    ```

3. Run the model

    For upscaling video:
    ```bash
    python test_onnx.py --model_path model.onnx --input path/to/video.mp4 --output path/to/output.mp4
    ```

    For upscaling images (untested):
    ```bash
    python test_onnx.py --model_path model.onnx --input example/lr/ --output example/sr/
    ```
    
    The test script supports both image and video inputs/outputs, similar to the PyTorch testing scripts. Additional arguments:
    - `--video`: Specify video codec for video output (e.g., 'h264_nvenc', 'libx264')
    - `--res`: Output video resolution
    - `--fps`: Output frame rate
    - `--presize`: Resize input before processing
    - `--providers`: ONNX Runtime execution providers (default: 'CUDAExecutionProvider,CPUExecutionProvider')


Original Paper
----------
[[Paper](https://arxiv.org/pdf/2203.13278.pdf)]

```bibtex
@article{zhang2022practical,
title={Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis},
author={Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Timofte, Radu and Van Gool, Luc},
journal={arXiv preprint},
year={2022}
}
```
