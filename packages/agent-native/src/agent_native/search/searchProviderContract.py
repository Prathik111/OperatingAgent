from abc import ABC, abstractmethod


class SearchProviderContract(ABC):

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> dict:
        pass