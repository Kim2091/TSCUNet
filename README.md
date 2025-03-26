# _Practical Single-Image and Temporal Upscaling via Swin-Conv-UNet_


Codes
---------

1. Single-Image inference

    ```bash
    python test_sisr.py --model_path pretrained_models/scunet_color_real_psnr.pth --input example/lr/ --output example/sr/ --depth 16
    ```

2. Temporal inference

    ```bash
    python test_vsr.py --model_path pretrained_models/2x_eula_anifilm_vsr.pth --input example/lr/ --output example/sr/ --depth 16
    ```
    Temporal models are curently not publicly available, and existing SCUNet models are not compatible with the temporal architecture.
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

Swin-Conv-UNet (SCUNet) denoising network
----------
<img src="figs/arch_scunet.png" width="900px"/> 

*The architecture of the proposed Swin-Conv-UNet (SCUNet) denoising network. SCUNet exploits the swin-conv (SC) block as
the main building block of a UNet backbone. In each SC block, the input is first passed through a 1×1 convolution, and subsequently is
split evenly into two feature map groups, each of which is then fed into a swin transformer (SwinT) block and residual 3×3 convolutional
(RConv) block, respectively; after that, the outputs of SwinT block and RConv block are concatenated and then passed through a 1×1
convolution to produce the residual of the input. “SConv” and “TConv” denote 2×2 strided convolution with stride 2 and 2×2 transposed
convolution with stride 2, respectively.*
