# agent/agent.py

import json
import os

from agent_native.context import Context
from agent_native.pipeline import Pipeline
from agent_native.llm.providerFactory import ProviderFactory
from agent_native.logger.consoleLogger import ConsoleLogger
from agent_native.logger.logLevel import LogLevel

class Agent:

    def __init__(self, config_path: str = "config.json"):

        self.config = self._load_config(config_path)

        self.agent_config = self.config["agent"]
        self.cost_config = self.config["cost_tracking"]

        # Create the configured LLM provider
        self.llm = ProviderFactory.create(
            self.config["llm"]
        )

        self.pipeline: Pipeline | None = None

        self.verbose = self.agent_config.get("verbose", False)

        self.logger= ConsoleLogger( level=LogLevel.DEBUG
        )

    # ---------------------------------------------------------

    def set_pipeline(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    # ---------------------------------------------------------

    def run(self, task: str) -> str:

        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline not set. Call agent.set_pipeline() before run()."
            )

        context = Context(task=task)

        iteration = 0

        max_iterations = self.agent_config.get(
            "max_iterations",
            5
        )

        while (
            not context.done
            and iteration < max_iterations
        ):

            self.pipeline.run(context)

            iteration += 1

        if self.verbose:
            self._print_summary(context)

        return context.result

    # ---------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        try:

            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)

            return self._resolve_env(config)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON in config file: {e}"
            )

    def _resolve_env(self, value):

        if isinstance(value, dict):
            return {
                k: self._resolve_env(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                self._resolve_env(v)
                for v in value
            ]

        if (
            isinstance(value, str)
            and value.startswith("${")
            and value.endswith("}")
        ):

            env_name = value[2:-1]

            env_value = os.getenv(env_name)

            if env_value is None:
                raise RuntimeError(
                    f"Environment variable '{env_name}' is not set."
                )

            return env_value

        return value

    # ---------------------------------------------------------

    def _print_summary(self, context: Context) -> None:

        if not self.cost_config.get("enabled"):
            return

        total_input = sum(
            metric["input"]
            for metric in context.metrics
        )

        total_output = sum(
            metric["output"]
            for metric in context.metrics
        )

        total_time = sum(
            metric["time"]
            for metric in context.metrics
        )

        print("\n========== EXECUTION SUMMARY ==========\n")

        for metric in context.metrics:

            print(
                f"[{metric['layer'].upper()}] "
                f"Input: {metric['input']} | "
                f"Output: {metric['output']} | "
                f"Time: {metric['time']}s"
            )

        print("\n--------------------------------------")

        print(
            f"Total Tokens : {total_input + total_output}"
        )

        print(
            f"Total Time   : {total_time:.3f}s"
        )

        print("\n======================================")