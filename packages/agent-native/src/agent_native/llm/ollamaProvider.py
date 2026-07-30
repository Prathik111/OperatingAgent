# agent/llm/ollamaProvider.py

import time
from typing import Dict, Any, List, Optional

import ollama

from agent_native.llm.llmProviderContract import LLMProviderContract


class OllamaProvider(LLMProviderContract):

    def __init__(self, config: Dict[str, Any]):
        self.model = config["model"]
        self.host = config.get("base_url")

        # Configure the client once
        self.client = ollama.Client(host=self.host)

    # ---------------------------------------------------------
    # Text Generation
    # ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0
    ) -> Dict[str, Any]:

        start = time.time()

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            think=False,
            options={
                "temperature": temperature
            }
        )

        return {
            "text": response["response"],
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
            "time": round(time.time() - start, 3)
        }

    # ---------------------------------------------------------
    # Chat / Tool Calling
    # ---------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:

        start = time.time()

        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=False,
            options={
                "temperature": temperature
            }
        )

        return {
            "message": response["message"],
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
            "time": round(time.time() - start, 3)
        }

    # ---------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return False