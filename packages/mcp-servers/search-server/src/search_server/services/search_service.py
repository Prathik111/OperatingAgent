"""Search service implementation for the search server package.

The search service is intentionally small and reusable: it indexes document
metadata into a lightweight in-memory catalogue and exposes query methods that
can be wired via thin tools.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchService:
    """In-memory document search service with simple indexing semantics."""

    logger: logging.Logger = field(default_factory=lambda: LOGGER)
    indices: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def index_documents(self, index_name: str, documents: list[dict[str, Any]], *, context: Any = None) -> dict[str, Any]:
        """Store documents under a named index and return the inserted count.

        Documents are deep-copied so callers cannot mutate indexed data through
        nested references they still hold.
        """

        self.indices[index_name] = [deepcopy(document) for document in documents]
        payload = {"index": index_name, "count": len(documents)}
        if context is not None:
            context.logger.info("search_indexed", index=index_name, count=len(documents))
        return payload

    def list_indices(self, *, context: Any = None) -> dict[str, Any]:
        """Return the currently registered search index names."""

        payload = {"indices": sorted(self.indices)}
        if context is not None:
            context.logger.info("search_indices", indices=payload["indices"])
        return payload

    def search(self, index_name: str, query: str, *, context: Any = None) -> dict[str, Any]:
        """Search a named index by document string content against the query term.

        Matches are deep-copied so the caller cannot mutate the stored index.
        """

        if index_name not in self.indices:
            raise KeyError(f"index '{index_name}' is not registered.")
        matches: list[dict[str, Any]] = []
        for document in self.indices[index_name]:
            document_text = str(document).lower()
            if query.lower() in document_text:
                matches.append(deepcopy(document))
        payload = {"index": index_name, "query": query, "matches": matches}
        if context is not None:
            context.logger.info("search_executed", index=index_name, matches=len(matches))
        return payload
