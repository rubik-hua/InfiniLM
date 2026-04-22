#!/usr/bin/env python3
"""使用 transformers 加载 Yi-6B 模型并进行推理测试"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <model_path>")
    sys.exit(1)

model_path = sys.argv[1]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = "How are you"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nPrompt: {prompt}")
print("Generating...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nOutput: {result}")
