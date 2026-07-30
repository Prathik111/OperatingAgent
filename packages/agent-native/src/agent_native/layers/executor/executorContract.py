# agent/layers/executor/executorContract.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ExecutorContract(ABC):
    """
    Executes an entire plan as one conversation.

    The executor is responsible for:
    - maintaining execution memory
    - deciding when to call tools
    - producing the final answer
    """

    @abstractmethod
    def execute(
        self,
        task: str,
        plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes the complete plan.

        Args:
            task:
                Original user task.

            plan:
                Planner output.

        Returns:
            {
                "result": str,
                "observations": [...],
                "metrics": {
                    "input": int,
                    "output": int,
                    "time": float
                }
            }
        """
        pass