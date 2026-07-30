from agent_native.backend.tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class CreateFileTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "createFile",
            "description": (
                "Create a new file at the specified path. "
                "If the parent directories do not exist, create them automatically. "
                "If the file already exists, overwrite its contents. "
                "Use this tool whenever the user asks to create, save, or write a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path of the file to create, "
                            "including the filename and extension."
                        )
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Text content to write into the file. "
                            "If omitted, an empty file will be created."
                        )
                    }
                },
                "required": ["path"]
            }
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = input_data["path"]
            content = input_data.get("content", "")

            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(path, "w", encoding="utf-8") as file:
                file.write(content)

            return {
                "result": {
                    "path": path,
                    "content_length": len(content)
                },
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "result": None,
                "success": False,
                "error": str(e)
            }