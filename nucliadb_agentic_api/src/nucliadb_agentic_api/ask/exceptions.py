"""Package exceptions"""

from nucliadb_models.search import KnowledgeboxFindResults, PreQueryResult


class KnowledgeBoxNotFound(Exception):
    pass


class ResourceNotFoundError(Exception):
    pass


class InvalidQueryError(Exception):
    """Raised when parsing a query containing an invalid parameter"""

    def __init__(self, param: str, reason: str):
        self.param = param
        self.reason = reason
        super().__init__(f"Invalid query. Error in {param}: {reason}")


class InternalParserError(ValueError):
    """Raised when parsing fails due to some internal error"""


class NoRetrievalResultsError(Exception):
    def __init__(
        self,
        main: KnowledgeboxFindResults | None = None,
        prequeries: list[PreQueryResult] | None = None,
        prefilters: list[PreQueryResult] | None = None,
    ):
        self.main_query = main
        self.prequeries = prequeries
        self.prefilters = prefilters


class AnswerJsonSchemaTooLong(Exception):
    pass


class IncompleteFindResultsError(Exception):
    pass


class NucliaDBError(Exception):
    """NucliaDB internal error raised on 500 and other NucliaDB errors"""

    pass
