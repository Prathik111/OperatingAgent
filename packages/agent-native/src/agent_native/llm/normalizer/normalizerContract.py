# agent/llm/normalizers/normalizerContract.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class NormalizerContract(ABC):

    # ---------- Chat ----------

    @abstractmethod
    def normalize_chat_request(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def normalize_chat_response(
        self,
        response: Any
    ) -> Dict[str, Any]:
        pass

    # ---------- Generate ----------

    @abstractmethod
    def normalize_generate_response(
        self,
        response: Any
    ) -> Dict[str, Any]:
        pass