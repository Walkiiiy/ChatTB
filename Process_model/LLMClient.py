"""
Simplified LLM Client class for single-turn conversations with local models.
Supports loading large models across multiple GPUs (e.g., 2×3090 24GB).
"""

import torch
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMClient:
    """
    A simplified client class for single-turn interactions with local models.
    Supports loading large models with full accuracy across multiple GPUs.
    """

    def __init__(self, 
                 model_path: Optional[str] = None,
                 torch_dtype: torch.dtype = torch.float16,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 top_p: float = 0.95,
                 trust_remote_code: bool = True,
                 max_context_tokens: int = 8192,
                 do_sample: bool = False):
        """
        Initialize the LLM client.

        Args:
            model_path: Path to the model.
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
        self.do_sample = do_sample

        # Model components
        self.tokenizer = None
        self.model = None

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
            raise RuntimeError("Tokenizer not loaded.")
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

    def _clean_response(self, response: str) -> str:
        """
        Clean the model response by removing any remaining artifacts or unwanted content.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned response containing only the model's actual content
        """
        if not response:
            return ""
        
        # Remove any leading/trailing whitespace
        cleaned = response.strip()
        
        # Remove common artifacts that might appear at the beginning
        artifacts_to_remove = [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
            "<|im_end|>",
            "assistant:",
            "Assistant:",
            "assistant",
            "Assistant"
        ]
        
        # Remove artifacts from the beginning
        for artifact in artifacts_to_remove:
            while cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # Remove any trailing end tokens
        if cleaned.endswith("<|im_end|>"):
            cleaned = cleaned[:-len("<|im_end|>")].strip()
        
        # Remove any remaining user content or system content that might have leaked through
        lines = cleaned.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like user or system content
            if (line.startswith('<|im_start|>user') or 
                line.startswith('<|im_end|>') or
                line.startswith('user:') or
                line.startswith('User:') or
                line.startswith('<|im_start|>system') or
                line.startswith('system:') or
                line.startswith('System:')):
                continue
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines).strip()
        
        return cleaned

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Single-turn chat with optional system prompt.
        Returns only the model's response content without prompt or history.
        
        Args:
            user_prompt: The user's input message
            system_prompt: Optional system instruction
            
        Returns:
            The model's response content only
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        if system_prompt:
            self._validate_prompt_length(system_prompt, user_prompt)
        else:
            self._validate_prompt_length("", user_prompt)

        # Build conversation history for single turn
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": user_prompt})

        # Apply chat template to get the formatted prompt
        text = self.tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,enable_thinking=False,
        )
        
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]  # Store the input length

        # Generation parameters
        gen_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0
        }

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Extract only the new tokens (generated part)
        new_tokens = outputs[0][input_length:]
        model_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up any remaining artifacts
        model_response = self._clean_response(model_response)
        
        return model_response

    def set_generation_params(self, 
                              max_new_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              top_p: Optional[float] = None,
                              do_sample: Optional[bool] = None) -> None:
        """Update generation parameters."""
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if do_sample is not None:
            self.do_sample = do_sample

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "torch_dtype": str(self.torch_dtype),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "is_loaded": self.is_loaded()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLMClient(
        model_path="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B"
    )
    print("Testing model:")
    print(llm.chat("What is the capital of France?"))