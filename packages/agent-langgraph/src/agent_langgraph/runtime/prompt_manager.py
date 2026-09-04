from pathlib import Path

from common.config import PromptConfig

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptManager:
    """
    Provides prompts for LLM-based nodes.

    Executor does not need a prompt because it performs
    deterministic tool execution.
    """

    def __init__(self, prompts: PromptConfig | str | Path) -> None:
        if isinstance(prompts, PromptConfig):
            self._paths = {
                "planner": prompts.planner_prompt,
                "verifier": prompts.verifier_prompt,
                "responder": prompts.responder_prompt,
            }
        else:
            prompt_dir = Path(prompts)
            self._paths = {
                "planner": prompt_dir / "planner.txt",
                "verifier": prompt_dir / "verifier.txt",
                "responder": prompt_dir / "responder.txt",
            }

    def _load(self, name: str) -> str:
        path = self._paths[name]

        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt file not found: {path}"
            )

        return path.read_text(encoding="utf-8")

    def planner(self) -> str:
        return self._load("planner")

    def verifier(self) -> str:
        return self._load("verifier")

    def responder(self) -> str:
        return self._load("responder")
