import json

from agent_native.context import Context
from agent_native.layers.layerContract import LayerContract
from agent_native.layers.evaluator.evaluatorContract import EvaluatorContract
from agent_native.llm.llmProviderContract import LLMProviderContract


class Evaluator(LayerContract, EvaluatorContract):

    def __init__(
        self,
        evaluator_config: dict,
        llm_provider: LLMProviderContract,
        logger
    ):
        self.llm = llm_provider
        self.logger = logger
        self.temperature = evaluator_config.get("temperature", 0)

    # ---------------------------------------------------------
    # EvaluatorContract
    # ---------------------------------------------------------

    def evaluate(self, task: str, result: dict) -> dict:

        prompt = self._build_prompt(task, result)

        self.logger.prompt(prompt)

        response = self.llm.generate(
            prompt,
            temperature=self.temperature
        )

        self.logger.llm_response(response)

        raw = response["text"].strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            parsed = json.loads(raw)

        except json.JSONDecodeError:

            parsed = {
                "done": False,
                "correct": False,
                "thoughts": "Failed to parse evaluator response.",
                "next_action": "Retry the previous step."
            }

        self.logger.debug("Evaluator Decision:")
        self.logger.debug(parsed)

        metrics = {
            "input": response["input_tokens"],
            "output": response["output_tokens"],
            "time": response["time"]
        }

        self.logger.metrics(metrics)

        return {

            "done": parsed.get("done", False),

            "correct": parsed.get("correct", False),

            "thoughts": parsed.get("thoughts", ""),

            "next_action": parsed.get("next_action"),

            "metrics": metrics

        }

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def _build_prompt(self, task: str, result: dict) -> str:

        return f"""
You are an evaluation agent.

Determine whether the task has been completed successfully.

Original Task:
{task}

Execution Result:
{result.get("result")}

Tool Observations:
{json.dumps(result.get("observations"), indent=2)}

Determine:

1. Is the result correct?
2. Is the original task fully completed?
3. If not, what should happen next?

Respond ONLY with valid JSON.

{{
    "done": true,
    "correct": true,
    "thoughts": "Short explanation.",
    "next_action": null
}}
"""

    # ---------------------------------------------------------
    # LayerContract
    # ---------------------------------------------------------

    def run(self, context: Context) -> None:

        self.logger.info("Evaluator started.")

        result = self.evaluate(

            context.task,

            {
                "result": context.result,
                "observations": context.observations
            }

        )

        context.done = result["done"]

        context.history.append({

            "plan": context.plan,

            "result": context.result,

            "thoughts": result["thoughts"],

            "next_action": result["next_action"]

        })

        context.add_metrics(

            "evaluator",

            result["metrics"]

        )

        self.logger.info("Evaluator finished.")