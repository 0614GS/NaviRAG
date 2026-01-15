from langchain.agents.middleware import SummarizationMiddleware, ToolRetryMiddleware, ModelFallbackMiddleware

from core.models.models import back_agent_model, summarize_model


def get_middlewares():
    summarize = SummarizationMiddleware(
        model=summarize_model,
        trigger=("tokens", 4000),
        keep=("messages", 20),
    )

    tool_retry = ToolRetryMiddleware(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
    )

    model_fallback = ModelFallbackMiddleware(
        back_agent_model,
    )

    return [summarize, tool_retry, model_fallback]
