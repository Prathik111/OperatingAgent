from abc import ABC, abstractmethod


class LoggerContract(ABC):

    @abstractmethod
    def iteration(self, current: int, maximum: int):
        pass

    @abstractmethod
    def pipeline(self, layer: str):
        pass

    @abstractmethod
    def prompt(self, prompt: str):
        pass

    @abstractmethod
    def llm_response(self, response):
        pass

    @abstractmethod
    def tool_call(self, name: str, arguments: dict):
        pass

    @abstractmethod
    def tool_result(self, result):
        pass

    @abstractmethod
    def metrics(self, metrics: dict):
        pass

    @abstractmethod
    def info(self, text: str):
        pass

    @abstractmethod
    def debug(self, text: str):
        pass

    @abstractmethod
    def trace(self, text: str):
        pass