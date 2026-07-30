from typing import List, Dict, Any

from .toolsContract import ToolContract

class ToolAdapter:
    """
    Converts internal ToolContract implementations into
    Ollama/Qwen-compatible tool definitions.
    """

    @staticmethod
    def to_qwen(tools: List[ToolContract]) -> List[Dict[str, Any]]:
        """
        Converts all registered tools into the format expected by
        Ollama's chat API.

        Returns:
        [
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        ]
        """

        converted = []

        for tool in tools:
            definition = tool.get_definition()

            converted.append({
                "type": "function",
                "function": {
                    "name": definition["name"],
                    "description": definition["description"],
                    "parameters": definition["parameters"]
                }
            })

        return converted