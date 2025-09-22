import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run language model inference')
    parser.add_argument('prompt', nargs='+', help='The prompt to generate text from')
    parser.add_argument('--model', '-m', choices=['phi2', 'phi4', 'phi5', 'flash', 'flash2'], default='phi2',
                       help='Choose model: phi2 or phi4 or phi5 (default: phi2)')
    
    args = parser.parse_args()
    
    # Join the prompt parts
    prompt = " ".join(args.prompt)
    
    # Map model choices to actual model names
    model_mapping = {

        
        'phi2': 'microsoft/phi-2',
        'phi4': 'microsoft/phi-4',
        'phi5': 'microsoft/phi-1_5',
        'flash': 'microsoft/phi-4-mini-flash-reasoning',
        'flash2': 'TheBloke/Flash-7B-GPTQ'
    }
    
    model_name = model_mapping[args.model]
    
    print(f"Using model: {model_name}")
    print(f"Prompt: {prompt}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # use MPS on Mac or CUDA if available
        dtype=torch.float32
    )

    # Tokenize and move inputs to the model's device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decode tokens into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip off the original prompt so you only see the answer
    clean_response = response[len(prompt):].strip()

    print("\n--- Response ---\n")
    print(clean_response)

if __name__ == "__main__":
    main()