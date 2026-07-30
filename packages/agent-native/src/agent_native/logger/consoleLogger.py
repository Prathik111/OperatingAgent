# agent/logger/consoleLogger.py

import json

from agent_native.logger.loggerContract import LoggerContract
from agent_native.logger.logLevel import LogLevel


class ConsoleLogger(LoggerContract):

    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.level = level

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _line(self):
        print("-" * 70)

    def _title(self, title: str):
        print()
        self._line()
        print(f"[{title.upper()}]")
        self._line()

    def _pretty(self, obj):

        if obj is None:
            print("None")
            return

        # Plain dict/list
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=4, default=str))
            return

        # Dataclass / class instance
        if hasattr(obj, "__dict__"):
            print(json.dumps(obj.__dict__, indent=4, default=str))
            return

        # Fallback
        print(obj)
    # ---------------------------------------------------------
    # General
    # ---------------------------------------------------------

    def info(self, text: str):
        print(f"[INFO] {text}")

    def debug(self, text: str):
        if self.level.value >= LogLevel.DEBUG.value:
            print(f"[DEBUG] {text}")

    def trace(self, text: str):
        if self.level.value >= LogLevel.TRACE.value:
            print(f"[TRACE] {text}")

    # ---------------------------------------------------------
    # Agent
    # ---------------------------------------------------------

    def iteration(self, current: int, maximum: int):
        self._title("Iteration")
        print(f"{current} / {maximum}")

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def pipeline(self, layer: str):
        self._title("Pipeline")
        print(f"Current Layer : {layer}")

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    def prompt(self, prompt: str):
        if self.level.value < LogLevel.DEBUG.value:
            return

        self._title("LLM Prompt")
        print(prompt)

    def llm_response(self, response):
        if self.level.value < LogLevel.DEBUG.value:
            return

        self._title("LLM Response")
        self._pretty(response)

    # ---------------------------------------------------------
    # Tool Calling
    # ---------------------------------------------------------

    def tool_call(self, name: str, arguments: dict):
        self._title("Tool Call")

        print(f"Tool : {name}")

        print("\nArguments:")

        self._pretty(arguments)

    def tool_result(self, result):
        self._title("Tool Result")

        self._pretty(result)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def metrics(self, metrics: dict):

        self._title("Metrics")

        print(f"Input Tokens  : {metrics.get('input', 0)}")
        print(f"Output Tokens : {metrics.get('output', 0)}")
        print(f"Execution Time: {metrics.get('time', 0)} s")