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
                # In TSCUNet, typically layer i processes window position i
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
        
        return x1

def convert_tscunet_to_onnx(model_path, onnx_path, clip_size=5, input_shape=None, dynamic=False, optimize=True):
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

    # Create dummy input
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)

    try:
        torch.onnx.export(
            export_model,              # model being run
            dummy_input,               # model input
            onnx_path,                 # where to save the model
            export_params=True,        # store the trained parameter weights
            opset_version=17,          # ONNX version
            do_constant_folding=True,  # optimize constants
            input_names=['input'],     
            output_names=['output'],   
            dynamic_axes=dynamic_axes, 
            verbose=False
        )
        
        print(f"Model successfully exported to {onnx_path}")
        
        # Verify the model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid!")
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
        not args.no_optimize
    )
