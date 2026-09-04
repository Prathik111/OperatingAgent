"""Tests for ``SearchService`` (search-server).

A small in-memory catalogue: index documents under a name, list indices, and
substring-search a named index. Fully hermetic (no I/O). Beyond the happy path,
the deep-copy isolation matters — a caller must not be able to mutate indexed
data through references it still holds, nor through returned matches.
"""

from __future__ import annotations

import pytest
from search_server.services.search_service import SearchService


@pytest.fixture
def service() -> SearchService:
    return SearchService()


# ---------------------------------------------------------------------------
# Indexing / listing
# ---------------------------------------------------------------------------


def test_index_documents_reports_count(service: SearchService) -> None:
    payload = service.index_documents("docs", [{"a": 1}, {"b": 2}])
    assert payload == {"index": "docs", "count": 2}


def test_list_indices_is_sorted(service: SearchService) -> None:
    service.index_documents("zeta", [])
    service.index_documents("alpha", [])
    assert service.list_indices()["indices"] == ["alpha", "zeta"]


def test_list_indices_empty_initially(service: SearchService) -> None:
    assert service.list_indices()["indices"] == []


def test_reindexing_replaces_previous_documents(service: SearchService) -> None:
    service.index_documents("docs", [{"body": "old"}])
    service.index_documents("docs", [{"body": "new"}])
    assert service.search("docs", "old")["matches"] == []
    assert len(service.search("docs", "new")["matches"]) == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_matches_substring_case_insensitively(service: SearchService) -> None:
    service.index_documents("docs", [{"title": "Hello World"}, {"title": "Goodbye"}])
    matches = service.search("docs", "hello")["matches"]
    assert matches == [{"title": "Hello World"}]


def test_search_returns_all_matches(service: SearchService) -> None:
    service.index_documents("docs", [{"t": "cat"}, {"t": "category"}, {"t": "dog"}])
    matches = service.search("docs", "cat")["matches"]
    assert len(matches) == 2


def test_search_missing_index_raises_key_error(service: SearchService) -> None:
    with pytest.raises(KeyError):
        service.search("nope", "anything")


# ---------------------------------------------------------------------------
# Deep-copy isolation
# ---------------------------------------------------------------------------


def test_mutating_source_after_index_does_not_change_index(service: SearchService) -> None:
    doc = {"id": "x", "body": "findme"}
    service.index_documents("docs", [doc])
    doc["body"] = "tampered"  # caller mutates the object it passed in

    matches = service.search("docs", "findme")["matches"]
    assert matches == [{"id": "x", "body": "findme"}]  # stored copy is intact


def test_mutating_returned_match_does_not_change_index(service: SearchService) -> None:
    service.index_documents("docs", [{"id": "x", "body": "findme"}])

    first = service.search("docs", "findme")["matches"]
    first[0]["body"] = "tampered"  # caller mutates the returned match

    again = service.search("docs", "findme")["matches"]
    assert again == [{"id": "x", "body": "findme"}]  # index unaffected
