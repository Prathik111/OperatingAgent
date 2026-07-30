# agent/llm/providerFactory.py

from agent_native.llm.llmProviderContract import LLMProviderContract


class ProviderFactory:

    @staticmethod
    def create(config: dict) -> LLMProviderContract:
        """
        Creates the configured LLM provider.

        Imports provider implementations lazily so optional provider
        dependencies (e.g., `ollama`) are not required unless selected
        in the runtime config.

        Args:
            config: The 'llm' section from config.json.

        Returns:
            An implementation of LLMProviderContract.
        """

        provider = config["provider"]

        if provider == "ollama":
            from agent_native.llm.ollamaProvider import OllamaProvider

            return OllamaProvider(config["ollama"])

        if provider == "groq":
            from agent_native.llm.groqProvider import GroqProvider

            return GroqProvider(config["groq"])

        raise ValueError(f"Unknown LLM provider '{provider}'")