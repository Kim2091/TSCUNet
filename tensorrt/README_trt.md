# TensorRT Guide

This guide helps you use TSCUNet with TensorRT for accelerated video upscaling.

### **Current Limitations:**
- FP32 precision only
- Static shapes required, must be multiples of 64 in each dimension
- Input videos must either:
  - Be padded to match engine dimensions (e.g., 720x480 → 768x512)
  - Fit exactly to required dimensions

## Setup Process

1. Set up VapourSynth following [pifroggi's guide](https://github.com/pifroggi/vapoursynth-stuff/blob/main/docs/vapoursynth-portable-setup-tutorial.md)
2. Download and extract `vsmlrt-windows-x64-tensorrt.[version].7z` from [vs-mlrt releases](https://github.com/AmusementClub/vs-mlrt/releases) to your `vs-plugins` directory
3. Get the model:
   - Download pre-converted ONNX from [releases](https://github.com/Kim2091/Kim2091-Models/releases), or
   - Convert your own using `convert_to_onnx_for_vsmlrt.py` (see script for detailed options)

## Usage

1. Build TensorRT engine using `trtexec`:
```bash
trtexec --onnx="tscunet_fp32.onnx" --optShapes=input:1x15x512x768 --saveEngine=tscunet_fp32.engine --builderOptimizationLevel=5 --useCudaGraph --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
```
    - You'll want to change the shape in the `--optShapes=input:1x15x512x768` section depending on the resolution of your input video
    - Please note that the shape has to be a multiple of 64. So if your input video is 720x540, you'd want to use `--optShapes=input:1x15x512x768`

2. Copy `vapoursynth_script.py` to your VapourSynth directory, then configure it with your video path and engine path

3. Open a Command Prompt window (NOT POWERSHELL) in your VapourSynth directory, then run a command like this. Customize the encoder settings as you wish:
```bash
vspipe -c y4m ".\vapoursynth_script.vpy" - | ffmpeg -i - -c:v hevc_nvenc -qp 0 -preset p5 -tune lossless "output.mkv"
```