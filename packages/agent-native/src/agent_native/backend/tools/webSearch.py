from agent_native.backend.tools.toolsContract import ToolContract
from typing import Dict, Any
import requests


class WebSearchTool(ToolContract):

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "webSearch",
            "description": (
                "Search the web for current information, news, documentation, "
                "tutorials, or general knowledge using SearXNG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of search results to return.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = input_data["query"]
            limit = input_data.get("limit", 5)

            response = requests.get(
                "http://localhost:8080/search",
                params={
                    "q": query,
                    "format": "json"
                },
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

            results = []

            for result in data.get("results", [])[:limit]:
                results.append({
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "content": result.get("content")
                })

            return {
                "result": {
                    "query": query,
                    "results": results
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