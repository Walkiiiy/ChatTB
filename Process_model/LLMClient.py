"""
LLM Client class for interacting with local models with full accuracy.
Supports loading large models across multiple GPUs (e.g., 2×3090 24GB).
"""

import torch
import logging
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


class LLMClient:
    """
    A client class for interacting with local models.
    Supports loading large models with full accuracy across multiple GPUs.
    """

    def __init__(self, 
                 model_path: Optional[str] = None,
                 torch_dtype: torch.dtype = torch.float16,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 top_p: float = 0.9,
                 trust_remote_code: bool = True,
                 max_context_tokens: int = 8192):
        """
        Initialize the LLM client.

        Args:
            model_path: Path to the Qwen model.
            torch_dtype: Data type for model weights (default: float16).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            trust_remote_code: Whether to trust remote code in model.
            max_context_tokens: Maximum context window size in tokens.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self.max_context_tokens = max_context_tokens

        # Model components
        self.tokenizer = None
        self.model = None
        self.history = []

        # Initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load the tokenizer and model with full accuracy (no quantization) + device_map=auto."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=self.trust_remote_code
            )

            # Load model without quantization for full accuracy
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",           # Distribute across GPUs
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True
            )

            self.logger.info("Model loaded successfully with full accuracy (no quantization)")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call _load_model() first.")
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return len(text) // 4

    def _validate_prompt_length(self, system_prompt: str, user_prompt: str) -> None:
        """Ensure prompt length fits inside the model's context window."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        prompt_tokens = self._count_tokens(full_prompt)
        total_estimated_tokens = prompt_tokens + self.max_new_tokens + 100

        if total_estimated_tokens > self.max_context_tokens:
            raise ValueError(
                f"Prompt too long for context window! "
                f"(tokens={total_estimated_tokens}, limit={self.max_context_tokens})"
            )

    def generate_response(self, 
                          user_prompt: str, 
                          system_prompt: Optional[str] = None,
                          clear_history: bool = False,
                          **generation_kwargs) -> str:
        """Generate a response from the LLM given a user prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if system_prompt:
            self._validate_prompt_length(system_prompt, user_prompt)
        else:
            self._validate_prompt_length("", user_prompt)

        if clear_history:
            self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": user_prompt})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_params = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "top_p": generation_kwargs.get("top_p", self.top_p),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = response[len(text):].strip() if text in response else response.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Single-turn chat with optional system prompt."""
        return self.generate_response(user_prompt, system_prompt, clear_history=True)

    def continue_chat(self, user_prompt: str) -> str:
        """Multi-turn chat continuing from previous history."""
        return self.generate_response(user_prompt, clear_history=False)

    def clear_history(self) -> None:
        self.history = []
        self.logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()

    def set_generation_params(self, 
                              max_new_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              top_p: Optional[float] = None) -> None:
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        return {
            "model_path": self.model_path,
            "torch_dtype": str(self.torch_dtype),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "is_loaded": self.is_loaded(),
            "history_length": len(self.history)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLMClient(
        model_path="/home/ubuntu/walkiiiy/ChatTB/Model/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/573fa3901e5799703b1e60825b0ec024a4c0f1d3"
        )  # 你需要改成自己的路径
    print("Testing model:")
    print(llm.chat("Hello, how are you?"))
