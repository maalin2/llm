import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model = model.to("cuda")

    prompt = "Hello, my name is"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=20)

    print(tokenizer.decode(outputs[0]))

main()
