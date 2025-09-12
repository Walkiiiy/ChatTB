import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型到GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",   # 自动分配显存
        torch_dtype=torch.bfloat16,  # 如果GPU支持BF16（A100/H100）
        trust_remote_code=True
    )

    # 构造输入
    prompt = "写一首关于秋天的诗。"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 推理
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    print("=== 模型输出 ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
