# agent/llm/normalizers/groqNormalizer.py

import json
from typing import Any, Dict, List

from agent_native.llm.normalizers.normalizerContract import NormalizerContract


class GroqNormalizer(NormalizerContract):

    # ---------------------------------------------------------
    # Internal -> Groq
    # ---------------------------------------------------------

    def normalize_chat_request(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        groq_messages = []

        for message in messages:

            role = message["role"]

            # -----------------------------
            # Assistant Tool Calls
            # -----------------------------

            if role == "assistant" and message.get("tool_calls"):

                tool_calls = []

                for call in message["tool_calls"]:

                    tool_calls.append({
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
                    "tool_calls": tool_calls
                })

            # -----------------------------
            # Tool Message
            # -----------------------------

            elif role == "tool":

                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"]
                })

            # -----------------------------
            # User / System / Assistant
            # -----------------------------

            else:
                groq_messages.append(message)

        return groq_messages

    # ---------------------------------------------------------
    # Groq -> Internal
    # ---------------------------------------------------------

    def normalize_chat_response(
        self,
        response: Any
    ) -> Dict[str, Any]:

        choice = response.choices[0].message
        usage = response.usage

        message = {
            "role": choice.role,
            "content": choice.content or ""
        }

        if choice.tool_calls:

            message["tool_calls"] = []

            for call in choice.tool_calls:

                message["tool_calls"].append({
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
            "message": message,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "time": None
        }

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    def normalize_generate_response(
        self,
        response: Any
    ) -> Dict[str, Any]:

        usage = response.usage

        return {
            "text": response.choices[0].message.content,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "time": None
        }