from agent_native.agent import Agent
from agent_native.pipeline import Pipeline

from agent_native.layers.planner import Planner
from agent_native.layers.executor import Executor
from agent_native.layers.evaluator import Evaluator

from agent_native.llm.providerFactory import ProviderFactory

from agent_native.backend.tools.webSearch import WebSearchTool
from agent_native.backend.tools.createFile import CreateFileTool
from agent_native.backend.tools.editFile import EditFileTool
from agent_native.backend.tools.viewFolder import ViewFolderTool

from dotenv import load_dotenv
import os
from pathlib import Path
    


def build_tools(config: dict) -> list:
    """
    Instantiate all enabled tools from config.json.
    """

    available_tools = {
        "createFile": CreateFileTool,
        "editFile": EditFileTool,
        "viewFolder": ViewFolderTool,
        "webSearch": WebSearchTool,
    }

    tools = []

    for tool_name in config["tools"]["enabled"]:

        tool_class = available_tools.get(tool_name)

        if tool_class is None:
            raise ValueError(f"Unknown tool '{tool_name}'")

        tools.append(tool_class())

    return tools


def build_pipeline(agent: Agent, tools: list) -> Pipeline:
    """
    Create the processing pipeline.
    """

    planner = Planner(
        planner_config=agent.config["layers"]["planner"],
        llm_provider=agent.llm,
        logger=agent.logger
    )

    executor = Executor(
        executor_config=agent.config["layers"]["executor"],
        llm_provider=agent.llm,
        tools=tools,
        logger=agent.logger
    )

    evaluator = Evaluator(
        evaluator_config=agent.config["layers"]["evaluator"],
        llm_provider=agent.llm,
        logger=agent.logger
    )

    return Pipeline([
        planner,
        executor,
        evaluator
    ])


def main():

    load_dotenv()

    # Resolve config path: prefer package-local `backend/config.json`,
    # fall back to repository-level `packages/config.json` if present.
    requested = "backend/config.json"

    package_root = Path(__file__).resolve().parent.parent
    candidate1 = package_root / requested
    candidate2 = Path.cwd() / "packages" / "config.json"

    if candidate1.exists():
        config_path = str(candidate1)
    elif candidate2.exists():
        config_path = str(candidate2)
    else:
        config_path = requested

    print(f"Using config: {config_path}")

    agent = Agent(
        config_path=config_path
    )

    agent.llm = ProviderFactory.create(
        agent.config["llm"]
    )

    tools = build_tools(agent.config)

    pipeline = build_pipeline(agent, tools)

    agent.set_pipeline(pipeline)

    task = (
        "Who are you?"
    )

    result = agent.run(task)

    print("\n========== FINAL RESULT ==========\n")
    print(result)


if __name__ == "__main__":
    main()