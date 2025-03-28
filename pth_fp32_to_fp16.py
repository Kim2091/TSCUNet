import torch
import argparse
import os
from collections import OrderedDict

def convert_to_fp16(input_model_path, output_model_path=None):
    """
    Convert a PyTorch model from FP32 to FP16
    
    Args:
        input_model_path (str): Path to the input FP32 model
        output_model_path (str, optional): Path to save the FP16 model. 
                                          If None, will use input path with '_fp16' suffix
    """
    print(f"Loading model from {input_model_path}")
    state_dict = torch.load(input_model_path, map_location='cpu')
    
    # Create a new OrderedDict for the half precision state dict
    new_state_dict = OrderedDict()
    
    # Convert to half precision
    for key, value in state_dict.items():
        if value.dtype == torch.float32:
            new_state_dict[key] = value.half()
        else:
            new_state_dict[key] = value
            
    # Determine output path if not specified
    if output_model_path is None:
        base, ext = os.path.splitext(input_model_path)
        output_model_path = f"{base}_fp16{ext}"
    
    # Save the half precision model
    torch.save(new_state_dict, output_model_path)
    print(f"Converted model saved to {output_model_path}")
    
    # Show memory savings
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)  # MB
    fp16_size = os.path.getsize(output_model_path) / (1024 * 1024)  # MB
    print(f"Original size: {original_size:.2f} MB")
    print(f"FP16 size: {fp16_size:.2f} MB")
    print(f"Memory reduction: {(1 - fp16_size/original_size) * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model from FP32 to FP16")
    parser.add_argument("--model", type=str, required=True, help="Path to the input FP32 model")
    parser.add_argument("--output", type=str, help="Path to save the FP16 model (optional)")
    args = parser.parse_args()
    
    convert_to_fp16(args.model, args.output)
