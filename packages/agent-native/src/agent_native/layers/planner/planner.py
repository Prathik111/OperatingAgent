# agent/layers/planner/planner.py

import json
import time

from agent_native.layers.layerContract import LayerContract
from agent_native.layers.planner.plannerContract import PlannerContract
from agent_native.llm.llmProviderContract import LLMProviderContract
from agent_native.context import Context


class Planner(LayerContract, PlannerContract):

    def __init__(self, planner_config: dict, llm_provider: LLMProviderContract,logger):
        self.llm = llm_provider
        self.temperature = planner_config.get("temperature", 0.5)
        self.logger = logger

    # ---------------------------
    # PlannerContract method
    # ---------------------------
    def plan(self, task: str, feedback: dict | None = None) -> dict:
        """
        Pure business logic. No Context dependency.
 
        Args:
            task: The user's task/goal.
            feedback: Optional dict from a previous failed attempt, e.g.
                {"thoughts": "...", "next_action": "..."}. When present,
                the planner revises the plan instead of regenerating it
                from scratch.
        """
        prompt = self._build_prompt(task, feedback)

        self.logger.prompt(prompt)

        response = self.llm.generate(prompt, temperature=self.temperature)

        self.logger.llm_response(response)

        raw = response['text'].strip()

        # Strip markdown fences if the model wraps its JSON despite instructions
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            parsed = json.loads(raw)
            plan_steps = parsed['plan']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[PLANNER ERROR] Failed to parse plan: {e}")
            print(f"[PLANNER ERROR] Raw response was: {response['text']}")
            plan_steps = []

        if not plan_steps:
            print(f"[PLANNER WARNING] Model returned an empty plan for task: {task}")

        self.logger.debug(plan_steps)
        metrics = {
            "input": response["input_tokens"],
            "output": response["output_tokens"],
            "time": response["time"]
        }

        self.logger.metrics(metrics)

        return {
            "plan": plan_steps,
            "metrics": metrics
        }


    def _build_prompt(self, task: str, feedback: dict | None = None) -> str:
        feedback_section = ""
 
        if feedback and (feedback.get("thoughts") or feedback.get("next_action")):
            feedback_section = f"""
                A previous attempt at this task did not fully succeed.
                
                Previous evaluator feedback:
                - Thoughts: {feedback.get("thoughts", "")}
                - Suggested next action: {feedback.get("next_action", "")}
                
                Revise the plan to address this feedback instead of repeating the same steps.
            """
 
        return f"""You are a planning agent. Break down the task into clear, actionable steps.
 
                Task: {task}
                {feedback_section}
                Respond ONLY in valid JSON, no markdown, no explanation, in this exact format:
                {{
                    "plan": [
                        {{"step": 1, "description": "..."}},
                        {{"step": 2, "description": "..."}},
                        ...
                    ]
            }}"""

    # ---------------------------
    # LayerContract method
    # ---------------------------
    def run(self, context: Context) -> None:
        """
        Pipeline adapter. Reads from and writes to shared context.
        """
        feedback = context.history[-1] if context.history else None
 
        result = self.plan(context.task, feedback=feedback)
 
        context.plan = result['plan']
        context.add_metrics("planner", result['metrics'])