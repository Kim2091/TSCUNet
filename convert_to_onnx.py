import torch
import torch.onnx
import argparse
import os
import numpy as np
import math
from models.network_tscunet import TSCUNet

class TSCUNetExportWrapper(torch.nn.Module):
    """Wrapper for TSCUNet"""
    def __init__(self, model):
        super(TSCUNetExportWrapper, self).__init__()
        self.model = model
        self.clip_size = model.clip_size
        self.scale = model.scale
        self.dim = model.dim
        self.residual = model.residual
        
        # Move required components into this wrapper
        self.m_head = model.m_head
        self.m_layers = model.m_layers
        if self.residual:
            self.m_res = model.m_res
        self.m_upsample = model.m_upsample
        self.m_tail = model.m_tail
    
    def forward(self, x):
        # Get dimensions without unpacking tensor to avoid tracer errors
        b = x.shape[0]
        t = x.shape[1]
        c = x.shape[2]
        h = x.shape[3]
        w = x.shape[4]
        
        # Calculate padding with fixed math (no numpy)
        mult = 64
        paddingH = ((h + mult - 1) // mult) * mult - h
        paddingW = ((w + mult - 1) // mult) * mult - w
        
        # Add extra padding for evaluation mode
        paddingH += 64
        paddingW += 64
        
        # Calculate padding on each side
        paddingLeft = paddingW // 2
        paddingRight = paddingW - paddingLeft
        paddingTop = paddingH // 2
        paddingBottom = paddingH - paddingTop
        
        # Reshape and pad the input - CRITICAL FIX: proper reshaping
        x_reshaped = x.reshape(b * t, c, h, w)  # Flatten batch and time dimensions
        x_padded = torch.nn.functional.pad(
            x_reshaped, 
            (paddingLeft, paddingRight, paddingTop, paddingBottom),
            mode='reflect'
        )
        
        # Process through head
        x_processed = self.m_head(x_padded)
        
        # Reshape back to include batch and time dimensions
        h_padded = h + paddingH
        w_padded = w + paddingW
        x = x_processed.reshape(b, t, self.dim, h_padded, w_padded)
        x1 = x

        # For TSCUNet, we need to handle the specific layer processing differently
        if t > 2:  # Only process if we have enough frames
            layer_idx = 0  # Track which layer we're using
            temp_outputs = []
            
            # Process through each window of 3 frames
            for i in range(t - 2):
                # Extract 3 consecutive frames
                frame_window = x1[:, i:i+3]
                
                # Reshape properly for the layer
                frame_window_reshaped = frame_window.reshape(b, 3 * self.dim, h_padded, w_padded)
                
                # Use the specific layer for this position
                layer_idx = i % len(self.m_layers)  # In case t-2 > number of layers
                layer_output = self.m_layers[layer_idx](frame_window_reshaped)
                temp_outputs.append(layer_output)
            
            # Stack outputs along time dimension
            if len(temp_outputs) > 1:
                x1 = torch.stack(temp_outputs, dim=1)  # Shape: [b, t-2, dim, h_padded, w_padded]
            else:
                x1 = temp_outputs[0].unsqueeze(1)  # For the case when t-2 = 1
            
            # Squeeze time dimension if it's 1
            if x1.shape[1] == 1:
                x1 = x1.squeeze(1)  # Shape: [b, dim, h_padded, w_padded]
            else:
                # If multiple time steps, take the center one
                x1 = x1[:, (x1.shape[1]-1)//2]  # Shape: [b, dim, h_padded, w_padded]
        else:
            # Handle edge case with fewer frames
            raise ValueError(f"Input requires at least 3 frames but got {t}")
        
        # Apply residual connection if model has it
        if self.residual:
            mid_frame = x[:, self.clip_size//2]  # Shape: [b, dim, h_padded, w_padded]
            x1 = x1 + self.m_res(mid_frame)
        
        # Upscale
        x1 = self.m_upsample(x1)
        
        # Final processing
        x1 = self.m_tail(x1)
        
        # Crop to original dimensions * scale
        h_out = h * self.scale
        w_out = w * self.scale
        padTop_scaled = paddingTop * self.scale
        padLeft_scaled = paddingLeft * self.scale
        
        x1 = x1[:, :, padTop_scaled:padTop_scaled+h_out, padLeft_scaled:padLeft_scaled+w_out]
        
        # CRITICAL CHANGE: Force output to be exactly 4D (B, C, H, W)
        # This ensures ONNX export matches PyTorch output shape
        x1 = x1.view(b, -1, h_out, w_out)  # Explicit reshaping instead of conditional squeeze
        
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

def convert_tscunet_to_onnx(model_path, onnx_path, clip_size=5, input_shape=None, dynamic=False, optimize=True, verify=True):
    """
    Convert a TSCUNet PyTorch model to ONNX format
    Args:
        model_path: Path to the PyTorch model state dict
        onnx_path: Output path for the ONNX model
        clip_size: Number of frames in the input sequence
        input_shape: Input shape tuple (batch, clip_size, channels, height, width)
        dynamic: Whether to use dynamic axes for the ONNX model
        optimize: Whether to optimize the model for export
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
    
    # Export the model
    print(f"Exporting model to ONNX: {onnx_path}")

    # Set default input shape if not provided
    if input_shape is None:
        # Use a smaller resolution for export to reduce memory usage
        height, width = 256, 256
        input_shape = (1, clip_size, 3, height, width)  # Ensure clip_size matches model
    
    print(f"Using input shape: {input_shape}")
    
    # Create dummy input with proper temporal dimension
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)

    try:
        torch.onnx.export(
            export_model,              
            dummy_input,               
            onnx_path,                 
            export_params=True,        
            opset_version=17,          
            do_constant_folding=True,  
            input_names=['input'],     
            output_names=['output'],   
            dynamic_axes=dynamic_axes, 
            verbose=False
        )
        
        print(f"Model successfully exported to {onnx_path}")

        # Run verification only once if verify=True
        if verify:
            verify_onnx_output(export_model, onnx_path, dummy_input)
        else:
            print("Skipping verification step")

        # Verify the model structure (this is different from output verification)
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure is valid!")
        except ImportError:
            print("ONNX package not installed. Skipping validation.")
        except Exception as e:
            print(f"ONNX validation error: {e}")
            
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
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}.onnx"
    
    # Get clip_size from model (avoid loading full model twice)
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
        not args.no_verify  # Pass verification flag
    )