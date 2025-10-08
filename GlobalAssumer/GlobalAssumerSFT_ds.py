#!/usr/bin/env python
"""
Using DeepSpeed for finetuning Global Assumer.
"""

import argparse
import os
import torch
import deepspeed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
    
from Modules.GlobalAssumerDataloader import GlobalAssumerDataloader
# (If your model class is elsewhere, import accordingly)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Global Assumer with DeepSpeed")

    # DeepSpeed configuration
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--save_interval", type=int, default=100, help="Steps between saving checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--local_rank", type=int, default=0)
    # Data settings
    parser.add_argument("--rules_json_path", type=str, default="data/rules.json")
    parser.add_argument("--schema_path", type=str, default="data/schema.json")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    return args

def load_hf_model_as_torch(model_name_or_path: str,
                           dtype: torch.dtype = torch.bfloat16,
                           device: str = "cpu",
                           trust_remote_code: bool = True):
    '''
    Load a Hugging Face model as a standard torch.nn.Module for DeepSpeed or manual training.
    '''
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=None,  # Don't spread across GPUs; let DeepSpeed handle it
        trust_remote_code=trust_remote_code
    )

    # Move to device manually if needed
    model = model.to(device)
    model.train()

    print(f"✅ Loaded {model_name_or_path} as torch.nn.Module ({model.__class__.__name__})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    return model, tokenizer

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    
    # Initialize model
    model, tokenizer = load_hf_model_as_torch(model_name_or_path=args.model)  # your model class should subclass torch.nn.Module
    model.train()


    # Load dataset
    dataloader = GlobalAssumerDataloader(
        tokenizer=tokenizer,
        max_prompt_length=2048,
        rules_json_path=args.rules_json_path,
        schema_path=args.schema_path,
        skip_no_rules=False,
        schema_rows=0,
        batch_size=args.batch_size,
    )
    train_dataloader = dataloader.load_dataloader()
    

    # DeepSpeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

            model_engine.backward(loss)
            model_engine.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

            # save checkpoint
            if global_step % args.save_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                client_sd = {"step": global_step}
                ckpt_id = f"step-{global_step}"
                model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
