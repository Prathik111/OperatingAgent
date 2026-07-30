from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LLMProviderContract(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        pass

    @abstractmethod
    def supports_json_mode(self) -> bool:
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        pass