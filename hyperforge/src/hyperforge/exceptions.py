class AutheticationException(Exception):
    pass


class NoAvailableAgents(Exception):
    pass


class MaxRetries(Exception):
    pass


class ModelRetry(Exception):
    pass


class OpenAIChatCompletionsError(Exception):
    pass


class NotSupportedbyLLMException(Exception):
    pass
