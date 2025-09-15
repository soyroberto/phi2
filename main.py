import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python llm.py 'your prompt here'")
        sys.exit(1)

    # Take everything after the program name as the prompt
    prompt = " ".join(sys.argv[1:])

    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # use MPS on Mac or CUDA if available
        dtype=torch.float32
    )

    # Tokenize and move inputs to the model’s device
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
