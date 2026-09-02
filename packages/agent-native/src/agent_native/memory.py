"""Things worth remembering after the run ends.

An agent that forgets everything at the end of a run has to be re-instructed
every morning. Two different things fix that, and they're kept separate on
purpose:

**Project instructions** are a file the *user* writes - `AGENT.md` in the working
folder. "Always run the tests with uv." "This repo uses tabs." It's read at the
start of every session and appended to the system prompt. The user owns it, the
agent never edits it, and reading it is the whole feature.

**Memories** are notes the *agent* writes, when the user tells it something worth
keeping. Each one has a kind - a preference, a plain fact, or a correction - and
lookup is by keyword. No embeddings: keyword matching is easy to explain, easy to
debug, and for a handful of notes it is not meaningfully worse. The interesting
failure of a memory system isn't imprecise recall, it's remembering something
that stopped being true, and no amount of vector search helps with that.

Recall is scored by how many of the query's words appear in the note, with a
nudge for notes used recently, so the ones that keep proving useful stay near the
top.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The file a user writes to give an agent standing instructions for a project.
PROJECT_INSTRUCTIONS_FILE = "AGENT.md"

#: Don't paste a whole book into the system prompt if someone writes one.
MAX_INSTRUCTIONS_CHARS = 8_000

#: Words too common to tell two memories apart.
_STOP_WORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "then", "there", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "me", "my"]
)


class MemoryKind:
    """What sort of note this is. Plain strings so they survive storage."""

    PREFERENCE = "preference"  # how the user likes things done
    FACT = "fact"              # something true about the project
    CORRECTION = "correction"  # something the agent got wrong once


VALID_KINDS = (MemoryKind.PREFERENCE, MemoryKind.FACT, MemoryKind.CORRECTION)


@dataclass
class Memory:
    """One remembered note."""

    text: str
    kind: str = MemoryKind.FACT
    session_id: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def score_for(self, words: set) -> int:
        """How many of `words` this note mentions. Zero means it isn't a match."""
        if not words:
            return 0
        mine = _words(self.text)
        return len(words & mine)


class MemoryStore:
    """Writes and looks up memories through whichever Database is in use."""

    def __init__(self, database: Any, redactor: Any = None) -> None:
        self._db = database
        # A note the agent keeps is read back into a future prompt, so a secret in
        # one would outlive the run that saw it. Redact on the way in. None means no
        # redaction (the runtime installs one; a bare store in a test stays literal).
        self._redactor = redactor

    async def remember(self, text: str, kind: str = MemoryKind.FACT, session_id: str = "") -> Memory:
        """Keep a note. An unrecognised kind is filed as a plain fact rather than refused."""
        cleaned = text.strip()
        if self._redactor is not None:
            cleaned = self._redactor.redact_text(cleaned)
        memory = Memory(
            text=cleaned,
            kind=kind if kind in VALID_KINDS else MemoryKind.FACT,
            session_id=session_id,
        )
        await self._db.save_memory(memory)
        return memory

    async def recall(self, query: str, session_id: str = "", limit: int = 5) -> list:
        """The notes that best match `query`, best first.

        Matching happens here rather than in SQL so both stores behave
        identically - the same query returns the same notes in the same order
        whether they came from a dict or from Postgres.
        """
        words = _words(query)
        scored = []
        for memory in await self._db.load_memories(session_id):
            score = memory.score_for(words)
            if score:
                scored.append((score, memory.last_used_at, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        found = [memory for _score, _used, memory in scored[:limit]]
        for memory in found:
            memory.last_used_at = datetime.now(UTC)
            await self._db.touch_memory(memory.id, memory.last_used_at)
        return found

    async def recent(self, session_id: str = "", limit: int = 5) -> list:
        """The handful most recently useful, for seeding a fresh session's prompt."""
        memories = await self._db.load_memories(session_id)
        memories.sort(key=lambda m: m.last_used_at, reverse=True)
        return memories[:limit]


def _words(text: str) -> set:
    """The meaningful lowercase words in some text."""
    return {
        word
        for word in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if len(word) > 2 and word not in _STOP_WORDS
    }


def read_project_instructions(working_directory: str) -> str:
    """The contents of AGENT.md in the working folder, or an empty string.

    Never raises. A missing file is the normal case, and an unreadable one is not
    worth failing a session over - the agent just starts without the extra
    instructions, which is exactly how it behaved before this existed.
    """
    if not working_directory:
        return ""
    try:
        path = Path(working_directory).expanduser() / PROJECT_INSTRUCTIONS_FILE
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, RuntimeError, ValueError):
        return ""
    if len(text) > MAX_INSTRUCTIONS_CHARS:
        text = text[:MAX_INSTRUCTIONS_CHARS] + "\n... [instructions truncated]"
    return text
