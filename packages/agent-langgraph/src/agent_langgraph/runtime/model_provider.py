from common.config import AgentConfig, LLMConfig
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr


def _ollama_base_url(value: str | None) -> str | None:
    """Normalize user-entered Ollama URLs to the server root.

    LangChain's Ollama client appends ``/api`` itself. Passing an API path here
    produces the confusing ``404 page not found`` response from Ollama.
    """
    if not value:
        return None
    return value.rstrip("/").removesuffix("/api") or None


class ModelProvider:
    """
    Provides the LLM used by the LangGraph nodes.

    A single model instance is shared across graph invocations.
    Nodes decide how to use the model; ModelProvider only owns
    model construction/access.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._model = self.create_chat_model(config.llm)

    @staticmethod
    def create_chat_model(config: LLMConfig) -> BaseChatModel:
        """
        Centralized model construction from an ``LLMConfig``.

        Shared by the graph nodes and the LLM judge so both resolve providers
        identically (ollama, groq, openai, anthropic). Replace this
        implementation with the model backend used by the project.
        """

        provider = config.provider.strip().lower()

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p,
                num_predict=config.max_tokens,
                base_url=_ollama_base_url(config.base_url),
                client_kwargs={"timeout": config.timeout_seconds},
                async_client_kwargs={"timeout": config.timeout_seconds},
            )

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
                model_kwargs={"top_p": config.top_p},
                api_key=SecretStr(config.api_key),
                base_url=(config.base_url or None),
                max_retries=config.max_retries,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=config.model,
                timeout=config.timeout_seconds,
                temperature=config.temperature,
                max_tokens_to_sample=config.max_tokens,
                top_p=config.top_p,
                api_key=SecretStr(config.api_key),
                base_url=(config.base_url or None),
                stop=None,
                max_retries=config.max_retries,
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=config.timeout_seconds,
                max_completion_tokens=config.max_tokens,
                api_key=SecretStr(config.api_key),
                base_url=(config.base_url or None),
                max_retries=config.max_retries,
            )

        raise NotImplementedError(
            f"unsupported LLM provider: {config.provider!r}"
        )

    def get_model(self) -> BaseChatModel:
        """Return the shared chat model."""
        return self._model
