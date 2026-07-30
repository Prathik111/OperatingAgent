# agent/llm/normalizers/ollamaNormalizer.py

import uuid
from typing import Any, Dict, List

from agent_native.llm.normalizers.normalizerContract import NormalizerContract


class OllamaNormalizer(NormalizerContract):

    # ---------------------------------------------------------
    # Internal -> Ollama
    # ---------------------------------------------------------

    def normalize_chat_request(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        ollama_messages = []

        for message in messages:

            role = message["role"]

            # ---------------------------------------
            # Assistant Tool Calls
            # ---------------------------------------

            if role == "assistant" and message.get("tool_calls"):

                tool_calls = []

                for call in message["tool_calls"]:

                    tool_calls.append({
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]
                        }
                    })

                ollama_messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls
                })

            # ---------------------------------------
            # Tool Message
            # ---------------------------------------

            elif role == "tool":

                ollama_messages.append({
                    "role": "tool",
                    "content": message["content"]
                })

            # ---------------------------------------
            # User / Assistant / System
            # ---------------------------------------

            else:

                ollama_messages.append(message)

        return ollama_messages

    # ---------------------------------------------------------
    # Ollama -> Internal
    # ---------------------------------------------------------

    def normalize_chat_response(
        self,
        response: Any
    ) -> Dict[str, Any]:

        message = response["message"]

        internal = {
            "role": message.role,
            "content": message.content or ""
        }

        if message.tool_calls:

            internal["tool_calls"] = []

            for call in message.tool_calls:

                internal["tool_calls"].append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": dict(call.function.arguments)
                    }
                })

        return internal

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    def normalize_generate_response(
        self,
        response: Any
    ) -> Dict[str, Any]:

        return {
            "text": response["response"]
        }