from abc import ABC, abstractmethod
from agent_native.context import Context

class LayerContract(ABC):
    
    @abstractmethod
    def run(self, context: Context) -> None:
        """
        Access and mutate the shared context directly.
        
        Args:
            context: Context - Shared state, mutated in place
        
        Returns:
            None
        """
        pass