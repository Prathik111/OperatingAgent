from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolContract(ABC):

    @abstractmethod
    def get_definition(self) -> str:
        """
        Returns a description of the tool.
        """
        pass

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the tool.

        Returns:
        {
            "success": bool,
            "message": str,
            "data": {}
        }
        """
        pass