# agent/layers/executor/executor.py

import json
import time

from agent_native.context import Context
from agent_native.layers.layerContract import LayerContract
from agent_native.layers.executor.executorContract import ExecutorContract
from agent_native.llm.llmProviderContract import LLMProviderContract
from agent_native.backend.tools.tool_adapter import ToolAdapter


class Executor(LayerContract, ExecutorContract):

    def __init__(
        self,
        executor_config: dict,
        llm_provider: LLMProviderContract,
        tools: list,
        logger
    ):
        self.llm = llm_provider
        self.temperature = executor_config.get("temperature", 0)

        self.tools = tools
        self.qwen_tools = ToolAdapter.to_qwen(tools)

        self.logger = logger
    # ---------------------------------------------------------
    # ExecutorContract
    # ---------------------------------------------------------

    def execute(
        self,
        task: str,
        plan: list
    ) -> dict:

        start = time.time()

        observations = []

        total_input = 0
        total_output = 0

        # Build ONE conversation
        messages = self._build_initial_messages(
            task,
            plan
        )
        tool_iterations = 0
        max_tool_iterations = 30

        while tool_iterations < max_tool_iterations:

            self.logger.prompt(messages)


            response = self.llm.chat(
                messages=messages,
                temperature=self.temperature,
                tools=self.qwen_tools
            )

            self.logger.llm_response(response)

            total_input += response["input_tokens"]
            total_output += response["output_tokens"]

            assistant = response["message"]

            messages.append(assistant)

            tool_calls = assistant.get("tool_calls", [])

            # Finished
            if not tool_calls:

                metrics = {
                    "input": total_input,
                    "output": total_output,
                    "time": round(time.time() - start, 3)
                }

                self.logger.metrics(metrics)

                return {

                    "result": assistant.get("content", ""),

                    "observations": observations,

                    "metrics": metrics
                }

            # Execute every requested tool
            for call in tool_calls:

                tool_iterations += 1
                
                observation = self._handle_tool_call(
                    call,
                    messages
                )

                observations.append(observation)
        raise RuntimeError(
            f"Maximum tool iterations ({max_tool_iterations}) exceeded."
        )
    # ---------------------------------------------------------
    # Build initial conversation
    # ---------------------------------------------------------

    def _build_initial_messages(
        self,
        task: str,
        plan: list
    ):

        formatted_plan = "\n".join(
            f"{step['step']}. {step['description']}"
            for step in plan
        )

        return [

            {
                "role": "system",
                "content": """
            You are an autonomous execution agent.

            You are given:

            - the user's original task
            - a plan created by a planner

            Execute the plan until the original task is fully completed.

            Rules:

            - Use tools whenever necessary.
            - Do not invent tool results.
            - Base decisions only on previous conversation and tool outputs.
            - Continue automatically after each tool call.
            - Only stop when the entire task is complete.
            - At the end, provide a concise summary.
            """
            },

            {
                "role": "user",
                "content":
                (
                    f"Task:\n{task}\n\n"
                    f"Plan:\n{formatted_plan}"
                )
            }

        ]
    # ---------------------------------------------------------
    # Handle a tool call
    # ---------------------------------------------------------

    def _handle_tool_call(
        self,
        call,
        messages: list
    ) -> dict:

        function = call["function"]

        tool_name = function["name"]
        arguments = function["arguments"]
        tool_call_id = call["id"]

        self.logger.tool_call(
            tool_name,
            arguments
        )

        tool = self._get_tool(tool_name)

        try:

            tool_result = tool.run(arguments)

        except Exception as e:

            tool_result = {
                "success": False,
                "error": str(e)
            }

        self.logger.tool_result(tool_result)

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "name": tool_name,
            "content": json.dumps(tool_result)
        })

        return {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "arguments": arguments,
            "result": tool_result
        }
    # ---------------------------------------------------------
    # Find tool
    # ---------------------------------------------------------

    def _get_tool(
        self,
        tool_name: str
    ):

        for tool in self.tools:

            definition = tool.get_definition()

            if definition["name"] == tool_name:
                return tool

        raise ValueError(
            f"Unknown tool '{tool_name}'"
        )
    # ---------------------------------------------------------
    # LayerContract
    # ---------------------------------------------------------

    def run(self, context: Context) -> None:

        self.logger.info("Executor started.")

        result = self.execute(
            task=context.task,
            plan=context.plan
        )

        context.result = result["result"]
        context.observations = result["observations"]

        context.add_metrics(
            "executor",
            result["metrics"]
        )

        self.logger.info("Executor finished.")