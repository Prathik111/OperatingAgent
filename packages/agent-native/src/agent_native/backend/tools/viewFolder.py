from agent_native.backend.tools.toolsContract import ToolContract
from typing import Dict, Any
import os


class ViewFolderTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "viewFolder",
            "description": (
                "List the contents of a directory. "
                "Use this tool when the user asks to view, browse, inspect, or list "
                "the files and subfolders inside a folder. "
                "This tool only returns the immediate contents of the directory and "
                "does not recursively list nested folders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path of the directory whose contents "
                            "should be listed."
                        )
                    }
                },
                "required": ["path"]
            }
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            path = input_data["path"]

            if not os.path.exists(path):
                return {
                    "result": None,
                    "success": False,
                    "error": f"Folder does not exist: {path}"
                }

            if not os.path.isdir(path):
                return {
                    "result": None,
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }

            items = []

            for item in os.listdir(path):
                full_path = os.path.join(path, item)

                items.append({
                    "name": item,
                    "type": "folder" if os.path.isdir(full_path) else "file"
                })

            return {
                "result": {
                    "path": path,
                    "items": items
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