import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='Run language model inference')
    parser.add_argument('prompt', nargs='+', help='The prompt to generate text from')
    parser.add_argument('--model', '-m', choices=['phi2', 'phi4', 'phi5', 'flash', 'phi31'], default='phi2',
                       help='Choose model: phi2, phi4, phi5, flash, or flash (default: phi2)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='auto', help='Device to run on')
    parser.add_argument('--dtype', choices=['float32', 'float16'], default=None, help='Precision type')               
    
    args = parser.parse_args()
    prompt = " ".join(args.prompt)
    
    model_mapping = {
        'phi2': 'microsoft/phi-2',
        'phi4': 'microsoft/phi-4',
        'phi5': 'microsoft/phi-1_5',
        'flash': 'microsoft/phi-4-mini-flash-reasoning'
    }
    model_name = model_mapping[args.model]

    # Device selection logic
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device


    # Dtype selection logic
    if args.dtype:
        dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    else:
        dtype = torch.float16 if device == 'cuda' else torch.float32

    print(f"Using model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto' if device != 'cpu' else None,
        torch_dtype=dtype
    ).to(device)

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_response = response[len(prompt):].strip()

    print("\n--- Response ---\n")
    print(clean_response)

if __name__ == "__main__":
    main()
