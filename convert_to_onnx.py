import torch
import torch.onnx
import argparse
import os
import numpy as np
import math
import onnx
from onnxconverter_common import float16
from models.network_tscunet import TSCUNet

class TSCUNetExportWrapper(torch.nn.Module):
    """Wrapper for TSCUNet that maintains closer compatibility with original implementation"""
    def __init__(self, model):
        super(TSCUNetExportWrapper, self).__init__()
        self.model = model
        self.clip_size = model.clip_size
        self.scale = model.scale
        self.dim = model.dim
        self.residual = model.residual
        self.sigma = model.sigma
        
        # Move required components into this wrapper
        self.m_head = model.m_head
        self.m_layers = model.m_layers
        if self.residual:
            self.m_res = model.m_res
        self.m_upsample = model.m_upsample
        self.m_tail = model.m_tail
        
        # Add sigma components if present
        if self.sigma:
            self.m_sigma = model.m_sigma
            self.m_sigma_tail = model.m_sigma_tail
        
    def forward(self, x):
        b, t, c, h, w = x.size()
        if t != self.clip_size:
            raise ValueError(
                f"input clip size {t} does not match model clip size {self.clip_size}"
            )

        # Calculate padding - using integer division instead of numpy for ONNX compatibility
        paddingH = -(-h // 64) * 64 - h  # Equivalent to ceil(h/64)*64 - h
        paddingW = -(-w // 64) * 64 - w  # Equivalent to ceil(w/64)*64 - w
        
        # Add extra padding for evaluation mode (always true for ONNX export)
        paddingH += 64
        paddingW += 64
        
        paddingLeft = math.ceil(paddingW / 2)
        paddingRight = math.floor(paddingW / 2)
        paddingTop = math.ceil(paddingH / 2)
        paddingBottom = math.floor(paddingH / 2)
        
        # Process through head
        x = (
            self.m_head(
                torch.nn.functional.pad(
                    x.view(-1, c, h, w),
                    (paddingLeft, paddingRight, paddingTop, paddingBottom),
                    mode='reflect'
                ).to(memory_format=torch.channels_last)
            )
            .to(memory_format=torch.contiguous_format)
            .view(b, -1, self.dim, h + paddingH, w + paddingW)
        )
        
        x1 = x
        
        # Process through temporal layers
        for layer in self.m_layers:
            temp = [None] * (t - 2)
            for i in range(t - 2):
                temp[i] = layer(
                    x1[:, i:i+3, ...].reshape(b, -1, h + paddingH, w + paddingW)
                    .to(memory_format=torch.channels_last)
                ).to(memory_format=torch.contiguous_format)
            x1 = torch.stack(temp, dim=1)
            t = x1.size(1)
        
        x1 = x1.squeeze(1).to(memory_format=torch.channels_last)
        
        # Apply residual if present
        if self.residual:
            x1 = x1 + self.m_res(
                x[:, self.clip_size//2, ...].to(memory_format=torch.channels_last)
            )
        
        # Upscale
        x1 = self.m_upsample(x1)
        
        # Apply sigma branch if present (for inference only)
        if self.sigma:
            sigma = self.m_sigma(x1)
            sigma = self.m_sigma_tail(sigma + x1).to(memory_format=torch.contiguous_format)
            sigma = sigma[
                ...,
                paddingTop * self.scale : paddingTop * self.scale + h * self.scale,
                paddingLeft * self.scale : paddingLeft * self.scale + w * self.scale,
            ]
        
        # Final processing
        x1 = self.m_tail(x1).to(memory_format=torch.contiguous_format)
        x1 = x1[
            ...,
            paddingTop * self.scale : paddingTop * self.scale + h * self.scale,
            paddingLeft * self.scale : paddingLeft * self.scale + w * self.scale,
        ]
        
        # Return appropriate outputs
        if self.sigma:
            return x1, sigma
        return x1

def verify_onnx_output(model, onnx_path, test_input, rtol=1e-2, atol=1e-3, save_outputs=False):
    """
    Verify ONNX model output against PyTorch model output
    
    Args:
        model: PyTorch model
        onnx_path: Path to the exported ONNX model
        test_input: Input tensor with shape (batch, clip_size, channels, height, width)
        rtol: Relative tolerance for output comparison
        atol: Absolute tolerance for output comparison
        save_outputs: Whether to save intermediate outputs for debugging
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Verify input shape
        if len(test_input.shape) != 5:
            raise ValueError(f"Expected 5D input (batch, time=5, channels, height, width), got shape {test_input.shape}")
        
        if test_input.shape[1] != 5:
            raise ValueError(f"TSCUNet requires exactly 5 frames for temporal processing, got {test_input.shape[1]} frames")
        
        if test_input.shape[1] != model.clip_size:
            raise ValueError(f"Input clip size {test_input.shape[1]} does not match model clip size {model.clip_size}")

        # Print input shape information
        print(f"\nInput shape details:")
        print(f"Input shape: {test_input.shape} (batch, time, channels, height, width)")
        
        # Get PyTorch output
        model.eval()
        with torch.inference_mode():
            torch_output = model(test_input).cpu().numpy()

        # Load and verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Print ONNX input details
        print("\nONNX model inputs:")
        for i, input_info in enumerate(ort_session.get_inputs()):
            print(f"  Input #{i}: name={input_info.name}, shape={input_info.shape}, type={input_info.type}")
        
        # Prepare input for ONNX Runtime
        ort_inputs = {
            ort_session.get_inputs()[0].name: test_input.cpu().numpy()
        }
        
        # Run ONNX model - get first output from the list
        onnx_outputs = ort_session.run(None, ort_inputs)
        onnx_output = onnx_outputs[0]  # Extract the first output tensor

        # Print detailed shape information
        print(f"\nOutput shape comparison:")
        print(f"PyTorch output shape: {torch_output.shape}")
        print(f"ONNX output shape: {onnx_output.shape}")

        # Compare outputs with more detailed statistics
        try:
            # Overall comparison
            np.testing.assert_allclose(
                torch_output,
                onnx_output,
                rtol=rtol,
                atol=atol
            )
            
            # Additional temporal-aware statistics
            abs_diff = np.abs(torch_output - onnx_output)
            max_diff = np.max(abs_diff)
            
            print("\nDetailed verification statistics:")
            print(f"✓ Maximum absolute difference: {max_diff:.6f}")
            print("✓ ONNX output verified against PyTorch output successfully.")
            
            if save_outputs and max_diff > rtol:
                # Save problematic regions for investigation
                worst_indices = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                print(f"\nLargest difference at index: {worst_indices}")
                print(f"PyTorch value: {torch_output[worst_indices]}")
                print(f"ONNX value: {onnx_output[worst_indices]}")
            
            return True
            
        except AssertionError as e:
            print(f"\n⚠ ONNX verification completed with warnings:")
            print(f"  {str(e)}")
            return False
            
    except ImportError:
        print("⚠ ONNX Runtime not installed. Skipping verification.")
        return False
    except Exception as e:
        print(f"❌ Error during ONNX verification: {str(e)}")
        return False

def convert_to_fp16(model_path, output_path=None):
    """Convert ONNX model to FP16 format"""
    if output_path is None:
        # Remove .onnx extension if present
        base_path = model_path[:-5] if model_path.endswith('.onnx') else model_path
        output_path = f"{base_path}_fp16.onnx"
    
    print(f"\nConverting model to FP16...")
    print(f"Loading ONNX model from {model_path}")
    
    try:
        onnx_model = onnx.load(model_path)
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model, 
            keep_io_types=True,
            op_block_list=['Pad', 'Resize']
        )
        
        print(f"Saving FP16 model to {output_path}")
        onnx.save(onnx_model_fp16, output_path)
        print("FP16 conversion completed successfully")
        return True
    except Exception as e:
        print(f"Error during FP16 conversion: {e}")
        return False

def convert_tscunet_to_onnx(model_path, onnx_path, clip_size=5, input_shape=None, dynamic=False, optimize=False, verify=True, fp16=False):
    """
    Convert a TSCUNet PyTorch model to ONNX format
    Args:
        model_path: Path to the PyTorch model state dict
        onnx_path: Output path for the ONNX model
        clip_size: Number of frames in the input sequence
        input_shape: Input shape tuple (batch, clip_size, channels, height, width)
        dynamic: Whether to use dynamic axes for the ONNX model
        optimize: Whether to optimize the model for export
        verify: Whether to verify the ONNX output
        fp16: Whether to also create an FP16 version
    """
    print(f"Loading PyTorch model from {model_path}")
    
    # Set device to CPU to reduce memory usage
    device = torch.device('cpu')
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize model with the loaded state
    model = TSCUNet(state=state_dict)
    model.eval()
    model = model.to(device)
    
    clip_size = model.clip_size
    scale = model.scale
    
    # Print model info
    print(f"Model clip size: {clip_size}")
    print(f"Model scale: {scale}x")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Set default input shape if not provided
    if input_shape is None:
        # Use a smaller resolution for export to reduce memory usage
        height, width = 256, 256
        input_shape = (1, clip_size, 3, height, width)
    
    print(f"Using input shape: {input_shape}")
    
    # Create dummy input with exactly 5 frames
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
    
    # Create export wrapper
    if optimize:
        export_model = TSCUNetExportWrapper(model)
    else:
        export_model = model
        
    export_model = export_model.to(device)
    
    # Define dynamic axes if requested
    dynamic_axes = None
    if dynamic:
        # Allow batch size, height and width to be dynamic
        dynamic_axes = {
            'input': {0: 'batch_size', 3: 'height', 4: 'width'},
            'output': {0: 'batch_size', 2: 'out_height', 3: 'out_width'}
        }
    
    # Modify the output path to include fp32/fp16
    base_path = os.path.splitext(onnx_path)[0]  # Strip any extension
    if base_path.endswith('.onnx'):  # Handle case where .onnx is part of the name
        base_path = base_path[:-5]
    fp32_path = f"{base_path}_fp32.onnx"

    # Export the model
    print(f"Exporting model to ONNX: {fp32_path}")

    try:
        torch.onnx.export(
            export_model,              
            dummy_input,               
            fp32_path,                # Use fp32_path instead of onnx_path
            export_params=True,        
            opset_version=17,          
            do_constant_folding=True,  
            input_names=['input'],     
            output_names=['output'],   
            dynamic_axes=dynamic_axes, 
            verbose=False
        )
        
        print(f"Model successfully exported to {fp32_path}")

        # Verify the model if requested
        if verify:
            verify_onnx_output(export_model, fp32_path, dummy_input)
        
        # Convert to FP16 if requested
        if fp16:
            fp16_path = f"{base_path}_fp16.onnx"
            convert_to_fp16(fp32_path, fp16_path)
            
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TSCUNet model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model")
    parser.add_argument("--output", type=str, help="Output path for ONNX model")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic axes")
    parser.add_argument("--no-optimize", action="store_true", help="Don't use optimized wrapper")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX output verification")
    parser.add_argument("--fp16", action="store_true", help="Also create FP16 version of the model")
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        # Strip any existing extension and use base name only
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}.onnx"
    else:
        # If output is specified, ensure we strip any extension
        base_name = os.path.splitext(args.output)[0]
        if base_name.endswith('.onnx'):  # Handle case where .onnx is part of the name
            base_name = base_name[:-5]
        args.output = f"{base_name}.onnx"
    
    # Get clip_size from model
    print("Loading model to determine clip_size...")
    temp_state = torch.load(args.model, map_location='cpu')
    temp_model = TSCUNet(state=temp_state)
    clip_size = temp_model.clip_size
    del temp_model, temp_state
    import gc
    gc.collect()
    
    # Define input shape
    input_shape = (args.batch, clip_size, 3, args.height, args.width)
    
    # Convert the model
    convert_tscunet_to_onnx(
        args.model, 
        args.output,
        clip_size,
        input_shape,
        args.dynamic,
        not args.no_optimize,
        not args.no_verify,
        args.fp16  # Pass FP16 flag
    )