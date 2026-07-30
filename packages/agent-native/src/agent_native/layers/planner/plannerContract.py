from abc import ABC, abstractmethod

class PlannerContract(ABC):
    
    @abstractmethod
    def plan(self, task: str) -> dict:
        """
        Generate a plan for the given task.
        
        Args:
            task: str - The user's task/goal
        
        Returns:
            dict:
                - plan: list of dicts
                    - step: int - Step number
                    - description: str - What needs to be done
                - metrics: dict
                    - input: int - Input tokens
                    - output: int - Output tokens
                    - time: float - Execution time in seconds
        """
        pass