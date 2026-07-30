from abc import ABC, abstractmethod

class EvaluatorContract(ABC):
    
    @abstractmethod
    def evaluate(self, task: str, result: dict) -> dict:
        """
        Review the execution result and decide next action.
        
        Args:
            task: str - The original user task
            result: dict - The result from the executor
                - result: str - Final output
                - observations: list - Raw observations from each step
        
        Returns:
            dict:
                - done: bool - Whether the task is complete
                - correct: bool - Whether the result is correct
                - thoughts: str - Reasoning about the result
                - next_action: str - What to do next if not done, None if done
                - metrics: dict
                    - input: int - Input tokens
                    - output: int - Output tokens
                    - time: float - Execution time in seconds
        """
        pass