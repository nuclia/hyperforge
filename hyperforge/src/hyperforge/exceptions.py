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


class CouldNotParse(Exception):
    def __init__(self, message: str | None = None, raw: str | None = None):
        self.message = message
        self.raw = raw


class ParseJsonSchemaException(Exception):
    pass
