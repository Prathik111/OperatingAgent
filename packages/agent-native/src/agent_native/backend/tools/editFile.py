from agent_native.backend.tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class EditFileTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "editFile",
            "description": (
                "Replace the entire contents of an existing file. "
                "Use this tool when the user asks to edit, modify, update, rewrite, "
                "or overwrite a file. The file must already exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path of the existing file to edit, "
                            "including the filename and extension."
                        )
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The complete new content that will replace the file's "
                            "existing contents."
                        )
                    }
                },
                "required": ["path", "content"]
            }
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = input_data["path"]
            content = input_data["content"]

            if not os.path.exists(path):
                return {
                    "result": None,
                    "success": False,
                    "error": f"File not found: {path}"
                }

            if not os.path.isfile(path):
                return {
                    "result": None,
                    "success": False,
                    "error": f"Path is not a file: {path}"
                }

            with open(path, "w", encoding="utf-8") as file:
                file.write(content)

            return {
                "result": {
                    "path": path
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