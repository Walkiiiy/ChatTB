"""
DeepSeek LLM Client class for interacting with DeepSeek API.
This class provides a simple interface to call DeepSeek's API for text generation.
"""

import requests
import json
import logging
from typing import List, Dict, Optional, Union
import time


class DeepSeekLLMClient:
    """
    A client class for interacting with the DeepSeek API.
    Supports various DeepSeek models through their API endpoint.
    """

    def __init__(self, 
                 api_key: str,
                 model: str = "deepseek-coder",
                 base_url: str = "https://api.deepseek.com/v1",
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 top_p: float = 0.9,
                 timeout: int = 60):
        """
        Initialize the DeepSeek LLM client.

        Args:
            api_key: DeepSeek API key
            model: Model name (e.g., "deepseek-coder", "deepseek-chat")
            base_url: Base URL for the API
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            timeout: Request timeout in seconds
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

        # Conversation history
        self.history = []

        # Validate API key
        if not self.api_key:
            raise ValueError("API key is required for DeepSeek API")

        self.logger.info(f"DeepSeek LLM client initialized with model: {self.model}")

    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """
        Make a request to the DeepSeek API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Generated response text
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }

        try:
            self.logger.debug(f"Making request to DeepSeek API: {url}")
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected API response format: {result}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise RuntimeError(f"DeepSeek API request failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            raise RuntimeError(f"Failed to parse DeepSeek API response: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during API call: {e}")
            raise

    def generate_response(self, 
                          user_prompt: str, 
                          system_prompt: Optional[str] = None,
                          clear_history: bool = False,
                          **generation_kwargs) -> str:
        """
        Generate a response from the DeepSeek API given a user prompt.

        Args:
            user_prompt: The user's input prompt
            system_prompt: Optional system prompt
            clear_history: Whether to clear conversation history
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        if clear_history:
            self.history = []

        # Prepare messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in self.history:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_prompt})

        # Override generation parameters if provided
        original_max_tokens = self.max_tokens
        original_temperature = self.temperature
        original_top_p = self.top_p

        if "max_tokens" in generation_kwargs:
            self.max_tokens = generation_kwargs["max_tokens"]
        if "temperature" in generation_kwargs:
            self.temperature = generation_kwargs["temperature"]
        if "top_p" in generation_kwargs:
            self.top_p = generation_kwargs["top_p"]

        try:
            response = self._make_request(messages)
            
            # Add to history
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": response})
            
            return response

        finally:
            # Restore original parameters
            self.max_tokens = original_max_tokens
            self.temperature = original_temperature
            self.top_p = original_top_p

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Single-turn chat with optional system prompt.

        Args:
            user_prompt: The user's input prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        return self.generate_response(user_prompt, system_prompt, clear_history=True)

    def continue_chat(self, user_prompt: str) -> str:
        """
        Multi-turn chat continuing from previous history.

        Args:
            user_prompt: The user's input prompt

        Returns:
            Generated response text
        """
        return self.generate_response(user_prompt, clear_history=False)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self.logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get a copy of the conversation history."""
        return self.history.copy()

    def set_generation_params(self, 
                              max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              top_p: Optional[float] = None) -> None:
        """
        Set generation parameters.

        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p

    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """Get information about the current model configuration."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "history_length": len(self.history)
        }

    def test_connection(self) -> bool:
        """
        Test the connection to the DeepSeek API.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.chat("Hello, this is a test message.")
            return len(response) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get API key from environment variable
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY environment variable")
        exit(1)
    
    # Initialize client
    client = DeepSeekLLMClient(api_key=api_key)
    
    # Test connection
    print("Testing DeepSeek API connection...")
    if client.test_connection():
        print("✅ Connection successful!")
        
        # Test chat
        print("\nTesting chat functionality:")
        response = client.chat("Hello, how are you?")
        print(f"Response: {response}")
        
        # Test with system prompt
        print("\nTesting with system prompt:")
        response = client.chat(
            "What is 2+2?", 
            system_prompt="You are a helpful math assistant."
        )
        print(f"Response: {response}")
        
    else:
        print("❌ Connection failed!")

