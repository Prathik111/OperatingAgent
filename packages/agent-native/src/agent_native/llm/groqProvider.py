# agent/llm/groqProvider.py

import json
import os
import time
from typing import Dict, Any, List, Optional

from groq import Groq

from agent_native.llm.llmProviderContract import LLMProviderContract


class GroqProvider(LLMProviderContract):

    def __init__(self, config):

        self.model = config["model"]

        api_key = os.getenv("GROQAPI")

        if api_key is None:
            raise RuntimeError(
                "GROQAPI environment variable not found."
            )

        self.client = Groq(api_key=api_key)

    # ---------------------------------------------------------
    # Text Generation
    # ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0
    ) -> Dict[str, Any]:

        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_object"
            }
        )

        usage = response.usage

        return {
            "text": response.choices[0].message.content,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
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

        # ---------------------------------------------
        # Convert internal format -> Groq format
        # ---------------------------------------------

        groq_messages = []

        for message in messages:

            # Ollama returns Pydantic objects
            if hasattr(message, "model_dump"):
                message = message.model_dump()

            role = message["role"]

            if role == "assistant" and message.get("tool_calls"):

                converted_calls = []

                for call in message["tool_calls"]:

                    converted_calls.append({
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": json.dumps(
                                call["function"]["arguments"]
                            )
                        }
                    })

                groq_messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": converted_calls
                })

            elif role == "tool":

                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"]
                })

            else:

                groq_messages.append(message)

        # ---------------------------------------------
        # Call Groq
        # ---------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,
            messages=groq_messages,
            temperature=temperature,
            tools=tools
        )

        choice = response.choices[0].message
        usage = response.usage

        # ---------------------------------------------
        # Convert Groq -> internal format
        # ---------------------------------------------

        internal = {
            "role": choice.role,
            "content": choice.content or ""
        }

        if choice.tool_calls:

            internal["tool_calls"] = []

            for call in choice.tool_calls:

                internal["tool_calls"].append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": json.loads(
                            call.function.arguments
                        )
                    }
                })

        return {
            "message": internal,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
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