# agent/search/searxngProvider.py

import requests

from agent_native.search.searchProviderContract import SearchProviderContract


class SearXNGProvider(SearchProviderContract):

    def __init__(self, config: dict):

        self.base_url = config["base_url"].rstrip("/")
        self.timeout = config.get("timeout", 20)
        self.default_results = config.get("max_results", 5)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int | None = None
    ) -> dict:

        if max_results is None:
            max_results = self.default_results

        response = requests.get(
            f"{self.base_url}/search",
            params={
                "q": query,
                "format": "json"
            },
            timeout=self.timeout
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("results", [])[:max_results]:

            results.append({

                "title": item.get("title", ""),

                "url": item.get("url", ""),

                "content": item.get("content", ""),

                "engine": item.get("engine", ""),

                "category": item.get("category", "")
            })

        return {

            "query": query,

            "count": len(results),

            "results": results
        }