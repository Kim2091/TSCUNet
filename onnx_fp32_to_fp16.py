import onnx
from onnxconverter_common import float16
import argparse
import time

def convert_to_fp16(model_path, output_path=None):
    if output_path is None:
        output_path = model_path.replace('.onnx', '_fp16.onnx')
    
    print(f"Loading model from {model_path}")
    start_time = time.time()
    
    onnx_model = onnx.load(model_path)
    print("Converting to FP16...")
    
    try:
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model, 
            keep_io_types=True,
            op_block_list=['Pad', 'Resize']
        )
        
        print(f"Saving to {output_path}")
        onnx.save(onnx_model_fp16, output_path)
        
        elapsed = time.time() - start_time
        print(f"Conversion completed in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--output", type=str, help="Output path for the FP16 model")
    
    args = parser.parse_args()
    convert_to_fp16(args.model, args.output)
