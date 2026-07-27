import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, cast
from uuid import uuid4

from nuclia.lib.nua_responses import Author, Image, Message
from nucliadb_models import (
    InputMessage,
    InputMessageContent,
    MessageType,
)
from nucliadb_models.conversation import Conversation
from nucliadb_models.resource import (
    Resource,
    ResourceField,
)
from nucliadb_models.search import FindOptions, FindRequest, KnowledgeboxFindResults
from nucliadb_sdk.v2 import NucliaDBAsync
from nucliadb_sdk.v2.exceptions import NotFoundError
from redis.asyncio import Redis

from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    Feedback,
    OAuthAuthenticateURL,
    StreamingChunk,
)
from hyperforge.models import (
    Answer,
    AnswerCitations,
    Chunk,
    Context,
    ExternalUsage,
    HistoryQuestionAnswer,
    MemoryConfig,
    Rule,
    Rules,
    Source,
    Step,
    TrackingInfo,
    Visualization,
)
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.server.cache import (
    Cache,
    CachedNucliaDBSource,
    CachedSessionQA,
    NoCache,
    ValkeyCache,
)

logger = logging.getLogger("arag.memory")

# NucliaDB storage
QUESTION_ANSWERS_FIELD: str = "qas"


def _qa_list_to_context_string(history: List[HistoryQuestionAnswer]) -> Tuple[str, int]:
    """Format a list of Q&A pairs into the prompt context string used by agents."""
    result = "".join(
        f"- Question: {qa.question}\n- Answer: {qa.answer}\n" for qa in history
    )
    return result, len(history)


def _qa_list_to_chat_messages(history: List[HistoryQuestionAnswer]) -> List[Message]:
    """Convert a list of Q&A pairs into the alternating User/Nuclia Message list used by LLMs."""
    return [
        msg
        for qa in history
        for msg in (
            Message(author=Author.USER, text=qa.question),
            Message(author=Author.NUCLIA, text=qa.answer),
        )
    ]


CONTEXT_FIELD: str = "context"
STEPS_FIELD: str = "steps"
USER_INFO_FIELD: str = "user_info"


class BaseSessionMemory:
    # Session ID
    id: str = "invalid"

    agent_id: str = ""
    workflow_id: str = ""
    account_id: str = ""
    kbid: Optional[str] = None

    # User information dictionary
    # Stored as JSON field
    user_info: Dict[str, str]

    # Configuration State
    rules: List[Union[Rule, str]]

    @classmethod
    def from_config(
        cls, config: MemoryConfig, agent_id: str, workflow_id: str, rules: Rules
    ):
        memory = cls(config, agent_id, workflow_id, NoCache())
        memory.rules = rules.rules
        return memory

    def __init__(
        self, config: MemoryConfig, agent_id: str, workflow_id: str, cache: Cache
    ):
        self.cache = cache
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.user_info = {}
        self.rules = []
        nucliadb_config = config.nucliadb
        self.kbid = nucliadb_config.kbid if nucliadb_config is not None else None

    def init(self, session: str):
        self.id = session

    async def set_source(self, source: Source):
        pass

    async def get_source(self, source_id: str) -> Source | None:
        return None

    def context_user_info(self) -> str:
        result = ""
        for key, value in self.user_info.items():
            result += f"- {key}: {value}\n"
        return result

    async def search_in_questions(self, question: str, all: bool):
        return KnowledgeboxFindResults(total=0, resources={})

    async def get_chat_history(self) -> List[Message]:
        return _qa_list_to_chat_messages(await self.qa_history())

    async def qa_history(self) -> list[HistoryQuestionAnswer]:
        return []

    async def context_history(self) -> Tuple[str, int]:
        return _qa_list_to_context_string(await self.qa_history())

    def start_question(
        self,
        question: str,
        actions: Optional[List[str]] = None,
        question_id: str | None = None,
        headers: Dict[str, str] = {},
        arguments: Dict[str, str] = {},
        streaming: bool = False,
        chat_history: Optional[List[HistoryQuestionAnswer]] = None,
    ) -> "QuestionMemory":
        return QuestionMemory(
            self,
            question,
            actions,
            question_id=question_id,
            headers=headers,
            arguments=arguments,
            streaming=streaming,
            chat_history=chat_history,
        )

    async def save(self, question: "QuestionMemory") -> None:
        pass


class NoMemorySessionMemory(BaseSessionMemory):
    def __init__(
        self, config: MemoryConfig, agent_id: str, workflow_id: str, cache: Cache
    ):
        self.cache = cache
        self.user_info = {}
        self.rules = []
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.debug = False

    def start_question(
        self,
        question: str,
        actions: Optional[List[str]] = None,
        question_id: str | None = None,
        headers: Dict[str, str] = {},
        arguments: Dict[str, str] = {},
        streaming: bool = False,
        chat_history: Optional[List[HistoryQuestionAnswer]] = None,
    ) -> "QuestionMemory":
        return QuestionMemory(
            self,
            question,
            actions,
            question_id=question_id,
            headers=headers,
            arguments=arguments,
            streaming=streaming,
            chat_history=chat_history,
        )

    async def save(self, question: "QuestionMemory") -> None:
        pass

    async def qa_history(self) -> list[HistoryQuestionAnswer]:
        return []

    async def set_source(self, source: Source):
        entry = CachedNucliaDBSource(
            cache=self.cache, agent_id=self.agent_id, source=source.id
        )
        await entry.set(source)

    async def get_source(self, source_id: str) -> Source | None:
        entry = CachedNucliaDBSource(
            cache=self.cache, agent_id=self.agent_id, source=source_id
        )
        return await entry.get()


class EphemeralSessionMemory(BaseSessionMemory):
    @classmethod
    def from_config(
        cls,
        config: MemoryConfig,
        agent_id: str,
        workflow_id: str,
        rules: Rules,
        client: Optional[Redis] = None,
    ):
        if client is None:
            memory = cls(config, agent_id, workflow_id, NoCache())
        else:
            memory = cls(config, agent_id, workflow_id, ValkeyCache(client=client))

        memory.rules = rules.rules
        return memory

    def __init__(
        self, config: MemoryConfig, agent_id: str, workflow_id: str, cache: Cache
    ):
        self.cache = cache
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.user_info = {}
        self.rules = []
        self.debug = False
        self.interactions: List[QuestionMemory] = []

    async def set_source(self, source: Source):
        entry = CachedNucliaDBSource(
            cache=self.cache, agent_id=self.agent_id, source=source.id
        )
        await entry.set(source)

    async def get_source(self, source_id: str) -> Source | None:
        entry = CachedNucliaDBSource(
            cache=self.cache, agent_id=self.agent_id, source=source_id
        )
        return await entry.get()

    def start_question(
        self,
        question: str,
        actions: Optional[List[str]] = None,
        question_id: str | None = None,
        headers: Dict[str, str] = {},
        arguments: Dict[str, str] = {},
        streaming: bool = False,
        chat_history: Optional[List[HistoryQuestionAnswer]] = None,
    ) -> "QuestionMemory":
        return QuestionMemory(
            self,
            question,
            actions,
            question_id=question_id,
            headers=headers,
            arguments=arguments,
            streaming=streaming,
            chat_history=chat_history,
        )

    async def save(self, question: "QuestionMemory") -> None:
        self.interactions.append(question)
        qa = HistoryQuestionAnswer(
            question=question.original_question,
            answer=question.final_answer or "",
        )
        await CachedSessionQA(self.cache, self.agent_id, self.id).append(qa)

    async def qa_history(self) -> list[HistoryQuestionAnswer]:
        cache_entry = CachedSessionQA(self.cache, self.agent_id, self.id)
        cached_qa = await cache_entry.get()
        if cached_qa is not None:
            return cached_qa

        return [
            HistoryQuestionAnswer(
                question=interaction.original_question,
                answer=interaction.final_answer or "",
            )
            for interaction in self.interactions
        ]


class SessionMemory(BaseSessionMemory):
    def __init__(
        self, config: MemoryConfig, agent_id: str, workflow_id: str, cache: Cache
    ):
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        if config.nucliadb is not None:
            self.url = config.nucliadb.url
            self.key = config.nucliadb.key
            self.kbid = config.nucliadb.kbid

            if config.nucliadb.internal:
                self.nucliadb_writer = NucliaDBAsync(
                    url=self.url.format(component="writer"),
                    api_key=self.key,
                    headers={"X-NUCLIADB-ROLES": "WRITER"},
                )
                self.nucliadb_reader = NucliaDBAsync(
                    url=self.url.format(component="reader"),
                    api_key=self.key,
                    headers={"X-NUCLIADB-ROLES": "READER"},
                )
                self.nucliadb_search = NucliaDBAsync(
                    url=self.url.format(component="search"),
                    api_key=self.key,
                    headers={"X-NUCLIADB-ROLES": "READER"},
                )
            else:
                self.nucliadb_writer = NucliaDBAsync(url=self.url, api_key=self.key)
                self.nucliadb_reader = NucliaDBAsync(url=self.url, api_key=self.key)
                self.nucliadb_search = NucliaDBAsync(url=self.url, api_key=self.key)

        self.cache = cache
        self.user_info = {}
        self.rules = []

    def init(self, session: str):
        self.id = session

    @property
    def debug(self):
        return logger.level

    @debug.setter
    def debug(self, debug: bool):
        if debug:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

    async def set_source(self, source: Source):
        if self.kbid:
            entry = CachedNucliaDBSource(
                cache=self.cache, agent_id=self.kbid, source=source.id
            )
            await entry.set(source)

    async def get_source(self, source_id: str) -> Source | None:
        if not self.kbid:
            return None

        entry = CachedNucliaDBSource(
            cache=self.cache, agent_id=self.kbid, source=source_id
        )
        return await entry.get()

    async def search_in_questions(self, question: str, all: bool):
        request = FindRequest(
            query=question,
            fields=["c/questions"],
            min_score=0.9,
            features=[FindOptions.SEMANTIC, FindOptions.KEYWORD],
        )
        if not all and self.id:
            request.resource_filters = [self.id]
        return await self.nucliadb_search.find(kbid=self.kbid, content=request)

    async def qa_history(self) -> list[HistoryQuestionAnswer]:
        if self.kbid is None:
            return []

        # Check if session Q&A is cached
        cache_entry = CachedSessionQA(self.cache, self.kbid, self.id)
        cached_qa = await cache_entry.get()
        if cached_qa is not None:
            return cached_qa

        # Retrieve from memory
        qas: list[HistoryQuestionAnswer] = []
        try:
            resource: Optional[
                Resource
            ] = await self.nucliadb_reader.get_resource_by_id(
                kbid=self.kbid, rid=self.id, query_params={"show": ["values"]}
            )
        except NotFoundError:
            resource = None

        if (
            resource is not None
            and resource.data is not None
            and resource.data.conversations is not None
        ):
            memory = resource.data.conversations.get(QUESTION_ANSWERS_FIELD)
            if memory is None or memory.value is None or memory.value.pages is None:
                raise NotImplementedError()
            else:
                for page_id in range(memory.value.pages):
                    page: Optional[
                        ResourceField
                    ] = await self.nucliadb_reader.get_resource_field(
                        kbid=self.kbid,
                        rid=self.id,
                        field_type="conversation",
                        field_id=QUESTION_ANSWERS_FIELD,
                        query_params={"page": page_id + 1},
                    )
                    if page is not None and page.value is not None:
                        conversation_value: Conversation = cast(
                            Conversation, page.value
                        )
                        question = None

                        for message in conversation_value.messages or []:
                            if message.type_ == MessageType.QUESTION:
                                question = message
                            elif (
                                message.type_ == MessageType.ANSWER
                                and question is not None
                                and question.content.text
                                and message.content.text
                            ):
                                qas.append(
                                    HistoryQuestionAnswer(
                                        question=question.content.text,
                                        answer=message.content.text,
                                    )
                                )
                        if len(qas) != 0:
                            await cache_entry.append_all(qas)

        return qas

    async def save(self, question: "QuestionMemory") -> None:
        # # Find or create memory resource (should already be created by create_session())
        # try:
        #     resource: Optional[
        #         Resource
        #     ] = await self.nucliadb_reader.get_resource_by_slug(
        #         kbid=self.kbid, slug=self.session
        #     )
        # except NotFoundError:
        #     resource = None

        # if resource is None:
        #     await self.nucliadb_writer.create_resource(
        #         kbid=self.kbid, content=CreateResourcePayload(slug=self.session)
        #     )
        #     resource = await self.nucliadb_reader.get_resource_by_id(
        #         kbid=self.kbid, slug=self.session
        #     )

        # # Save steps
        # steps_message = []
        # for step in self.steps:
        #     steps_message.append(
        #         InputMessage(
        #             ident=uuid4().hex,
        #             who=step.module,
        #             type=MessageType.UNSET,
        #             content=InputMessageContent(
        #                 text=step.markdown(), format=MessageFormat.KEEP_MARKDOWN
        #             ),
        #         )
        #     )

        # await self.nucliadb_writer.add_conversation_message(
        #     kbid=self.kbid,
        #     rid=self.session,
        #     slug=STEPS_FIELD,
        #     content=steps_message,
        # )

        # # Save context
        # contexts_message = []
        # for context in self.contexts:
        #     contexts_message.append(
        #         InputMessage(
        #             ident=uuid4().hex,
        #             who="",
        #             type=MessageType.UNSET,
        #             content=InputMessageContent(
        #                 text="", format=MessageFormat.KEEP_MARKDOWN
        #             ),
        #         )
        #     )

        # await self.nucliadb_writer.add_conversation_message(
        #     kbid=self.kbid,
        #     rid=self.session,
        #     slug=CONTEXT_FIELD,
        #     content=contexts_message,
        # )

        # Save Q & A
        if question.final_answer:
            if self.kbid:
                qa = HistoryQuestionAnswer(
                    question=question.original_question, answer=question.final_answer
                )
                await CachedSessionQA(self.cache, self.kbid, self.id).append(qa)

            content = [
                InputMessage(
                    who="user",
                    to=["agent"],
                    timestamp=question.started_at,
                    ident=f"q-{question.original_question_uuid}",
                    type=MessageType.QUESTION,
                    content=InputMessageContent(text=question.original_question),
                ),
                InputMessage(
                    who="agent",
                    to=["user"],
                    timestamp=datetime.now(timezone.utc),
                    ident=f"a-{question.original_question_uuid}",
                    type=MessageType.ANSWER,
                    content=InputMessageContent(text=question.final_answer),
                ),
            ]
            await self.nucliadb_writer.add_conversation_message(
                kbid=self.kbid,
                rid=self.id,
                field_id=QUESTION_ANSWERS_FIELD,
                content=content,
            )


class QuestionMemory:
    session: BaseSessionMemory

    headers: Dict[str, str]
    arguments: Dict[str, str]

    # Main RAG Block
    started_at: datetime
    original_actions: List[str]
    original_question: str
    original_question_uuid: str
    final_answer: Optional[str] = None
    final_answer_citations: Optional[AnswerCitations] = None
    final_answer_urls: List[str]
    answers: list[tuple[str, Optional[AnswerCitations]]]
    generated_texts: Dict[str, str]
    # Data visualizations generated
    data_visualizations: List[Visualization]

    # Whether after generation, we consider the question answered (for example we might have generated a response but the response was "not enough data to answer this")
    is_answered: bool = False

    # Callback information to return to user
    callback_fn: Optional[Callable[[AragAnswer], Awaitable[None]]] = None
    feedback_fn: Optional[
        Callable[[Feedback], Awaitable[UserToAgentInteraction | None]]
    ] = None
    oauth_fn: Optional[Callable[[OAuthAuthenticateURL], Awaitable[None]]] = None
    oauth_callback_fn: Optional[Callable[[str, str], Awaitable[str | None]]] = None

    # Short term memory
    steps: List[Step]
    contexts: List[Context]
    agent_contexts: Dict[str, Dict[str, List[Context]]]

    # Flow controls
    restart: bool = False
    secure: Optional[bool] = None
    streaming: bool = False

    future_questions: OrderedDict[str, str]
    context_questions: OrderedDict[str, str]
    actual_question: Optional[str] = None
    actual_question_uuid: Optional[str] = None

    generation_rules: OrderedDict[str, str]
    actual_action: Optional[str] = None
    actual_action_uuid: Optional[str] = None

    def __init__(
        self,
        session: BaseSessionMemory,
        question: str,
        actions: Optional[List[str]] = None,
        question_id: str | None = None,
        headers: Dict[str, str] | None = None,
        arguments: Dict[str, str] | None = None,
        streaming: bool = False,
        chat_history: Optional[List[HistoryQuestionAnswer]] = None,
    ):
        self.session = session
        self.started_at = datetime.now(timezone.utc)

        # Client-managed chat history. When set (even to an empty list), overrides
        # server-side session history for agents that use previous Q&A context
        # (rephrase, summarize, smart, etc.). None means "not set — use server-side
        # history". [] means "override with no history". Intended for ephemeral
        # sessions where the client is responsible for maintaining conversation state.
        # Note: search_in_questions() performs semantic search over NucliaDB-stored
        # conversation history and is NOT affected by this field. The HistoricalAgent
        # uses that method and therefore does not benefit from client-managed history.
        self._client_chat_history: Optional[List[HistoryQuestionAnswer]] = chat_history

        # Start of a new question by the user
        self.original_question = question
        if actions is not None:
            self.original_actions = actions
        self.restart = True
        if not question_id:
            question_id = uuid4().hex
        self.original_question_uuid = question_id

        self.headers = headers if headers is not None else {}
        self.arguments = arguments if arguments is not None else {}
        self.streaming = streaming
        self.contexts = []
        self.steps = []

        self.context_questions = OrderedDict()
        self.original_actions = []
        self.future_questions = OrderedDict()
        self.answers: list[tuple[str, Optional[AnswerCitations]]] = []
        self.generated_texts = {}
        self.data_visualizations = []
        self.final_answer_urls = []
        self.agent_contexts = {}

        self.set_actual_question(question, question_id)

    def get_session_id(self) -> str:
        """Returns the session ID for the current question. The session ID is a unique identifier that is shared across all questions and interactions that belong to the same session. This can be used to group related interactions together, and to keep track of the conversation history in a coherent way."""
        return self.session.id

    def get_agent_id(self) -> str:
        """Returns the agent ID for the current question. The agent ID is a unique identifier that is shared across all questions and interactions that belong to the same agent. This can be used to group related interactions together, and to keep track of the conversation history in a coherent way."""
        return self.session.agent_id

    def get_workflow_id(self) -> str:
        """Returns the workflow ID for the current question. The workflow ID is a unique identifier that is shared across all questions and interactions that belong to the same workflow. This can be used to group related interactions together, and to keep track of the conversation history in a coherent way."""
        return self.session.workflow_id

    def get_account_id(self) -> str:
        """Returns the account ID for the current question."""
        return self.session.account_id

    def context_user_info(self) -> str:
        """Returns a string with user information that can be used in the context of the agent. This can include information such as user preferences, user history, or any other relevant information about the user that can help the agent to generate a more personalized and accurate response."""
        return self.session.context_user_info()

    def get_rules(self) -> list[Rule | str]:
        """Returns the rules for the current question. The rules are a unique identifier that is shared across all questions and interactions that belong to the same set of rules. This can be used to group related interactions together, and to keep track of the conversation history in a coherent way."""
        return self.session.rules

    async def search_in_questions(
        self, question: str, all: bool = False
    ) -> KnowledgeboxFindResults:
        """Searches for similar questions in the conversation history. This can be used to find relevant information that has been previously discussed in the conversation, and to provide a more accurate and personalized response."""
        return await self.session.search_in_questions(question, all)

    def user_info(self) -> Dict[str, str]:
        """Returns a string with user information that can be used in the context of the agent. This can include information such as user preferences, user history, or any other relevant information about the user that can help the agent to generate a more personalized and accurate response."""
        return self.session.user_info

    async def set_session_source(self, source: Source):
        """Sets the source of the session. This can be used to keep track of where the information in the conversation is coming from, and to provide more context to the agent when generating a response."""
        return await self.session.set_source(source)

    async def get_session_source(self, source_id: str) -> Optional[Source]:
        """Gets the source of the session. This can be used to keep track of where the information in the conversation is coming from, and to provide more context to the agent when generating a response."""
        return await self.session.get_source(source_id)

    async def context_history(self) -> Tuple[str, int]:
        """Returns a string with the context history of the conversation. This can include information such as previous questions and answers, relevant information that has been previously discussed in the conversation, or any other relevant information that can help the agent to generate a more accurate and personalized response.

        When the client sets chat_history in the request (even to an empty list), it overrides any server-side session history. None means "not set — use server-side history"."""
        if self._client_chat_history is not None:
            return _qa_list_to_context_string(self._client_chat_history)
        return await self.session.context_history()

    async def get_chat_history(self) -> list[Message]:
        """Returns a list of tuples with the chat history of the conversation. Each tuple contains a question and an answer. This can be used to keep track of the conversation history in a more structured way, and to provide more context to the agent when generating a response.

        When the client sets chat_history in the request (even to an empty list), it overrides any server-side session history. None means "not set — use server-side history"."""
        if self._client_chat_history is not None:
            return _qa_list_to_chat_messages(self._client_chat_history)
        return await self.session.get_chat_history()

    def stats(self):
        return {
            "session": self.session,
            "contexts": [context.stats() for context in self.contexts],
            "context_questions": self.context_questions,
            "original_question": self.original_question,
            "original_actions": self.original_actions,
            "final_answer": self.final_answer,
        }

    async def save_context(self, flow_id: str, context: Context):
        context.original_question_uuid = self.original_question_uuid
        context.actual_question_uuid = self.actual_question_uuid
        self.contexts.append(context)
        if self.agent_contexts.get(flow_id) is None:
            self.agent_contexts[flow_id] = {}
        if self.agent_contexts[flow_id].get(context.agent_id) is None:
            self.agent_contexts[flow_id][context.agent_id] = []
        self.agent_contexts[flow_id][context.agent_id].append(context)
        if self.callback_fn is not None:
            await self.callback_fn(AragAnswer(context=context))

    async def save_image_urls(self, image_urls: List[str]):
        # We need to extend but deduplicate bedore extending, taking into account that we only check the part before ?eph-token
        # TODO: in the next iteration we should handle this better, i.e. savin the token separately
        existing_urls = {url.split("?eph-token")[0] for url in self.final_answer_urls}
        new_urls = [
            url for url in image_urls if url.split("?eph-token")[0] not in existing_urls
        ]
        self.final_answer_urls.extend(new_urls)

    def get_agent_contexts(self, flow_id: str, agent_id: str) -> List[Context]:
        flow_contexts = self.agent_contexts.get(flow_id, {})
        return flow_contexts.get(agent_id, [])

    def get_agent_answer_summaries(self, flow_id: str, agent_id: str) -> List[str]:
        flow_contexts = self.agent_contexts.get(flow_id, {})
        contexts = flow_contexts.get(agent_id, [])
        return [
            context.summary for context in contexts if context.summary.strip() != ""
        ]

    def list_contexts_markdown(self) -> list[str]:
        contexts_str = []
        for context in self.contexts:
            result = ""
            if context.citations_id is not None:
                if context.title:
                    result += f"## [{context.citations_id}] {context.title}\n\n"
                else:
                    result += f"## [{context.citations_id}]\n\n"
            else:
                if context.title:
                    result += f"## {context.title}\n\n"
            result += f"{context.context_markdown()}"
            contexts_str.append(result)
        return contexts_str

    def list_chunks_markdown(self) -> list[str]:
        chunks_str = []
        for context in self.contexts:
            for chunk in context.chunks:
                result = ""
                if context.citations_id is not None:
                    if chunk.title:
                        result += f"## [{context.citations_id}] {chunk.title}\n\n"
                    else:
                        result += f"## [{context.citations_id}]\n\n"
                else:
                    if chunk.title:
                        result += f"## {chunk.title}\n\n"
                result += f"{chunk.text}\n\n"
                chunks_str.append(result)
        return chunks_str

    def contexts_markdown(self) -> str:
        """
        Returns the concatenated contexts as a single string. Includes full context (i.e: all the chunk texts)
        """
        return "\n\n".join(self.list_contexts_markdown())

    def list_contexts_minimal(
        self,
    ) -> list[str]:
        contexts_str = []
        for context in self.contexts:
            result = ""
            if context.citations_id is not None:
                if context.title:
                    result += f"## [{context.citations_id}] {context.title}\n\n"
                else:
                    result += f"## [{context.citations_id}]\n\n"
            else:
                if context.title:
                    result += f"## {context.title}\n\n"
            if context.summary.strip() != "":
                result += f"{context.answer_summary_markdown()}"
            else:
                result += f"{context.context_markdown()}"
            contexts_str.append(result)
        return contexts_str

    def contexts_minimal(self) -> str:
        """
        Returns the concatenated minimal contexts as a single string.
        Minimal contexts include summaries if available, otherwise full context (i.e: the chunk texts)
        """
        return "\n\n".join(self.list_contexts_minimal())

    def get_prompt_texts(self) -> List[str]:
        prompts = []
        for context in self.contexts:
            for prompt in context.prompts:
                prompts.append(prompt.render())
        return prompts

    async def add_generated_text(self, generation_id: str, generated_text: str):
        self.generated_texts[generation_id] = generated_text
        if self.callback_fn is not None:
            await self.callback_fn(AragAnswer(generated_text=generated_text))

    async def add_step(
        self,
        step_module: str,
        step_title: str,
        step_agent_path: str,
        timeit: float,
        input_nuclia_tokens: Optional[float] = None,
        output_nuclia_tokens: Optional[float] = None,
        step_value: Optional[str] = None,
        step_reason: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        external_usage: Optional[List[ExternalUsage]] = None,
    ):
        new_step = Step(
            original_question_uuid=self.original_question_uuid,
            actual_question_uuid=self.actual_question_uuid,
            module=step_module,
            title=step_title,
            agent_path=step_agent_path,
            value=step_value,
            reason=step_reason,
            timeit=timeit,
            input_nuclia_tokens=input_nuclia_tokens,
            output_nuclia_tokens=output_nuclia_tokens,
            error=error,
            metadata=metadata,
            external_usage=external_usage,
        )
        self.steps.append(new_step)
        if self.callback_fn is not None:
            await self.callback_fn(AragAnswer(step=new_step))

    def set_actual_question(self, question: str, uuid: Optional[str] = None):
        self.actual_question = question
        if uuid is None:
            uuid = uuid4().hex
        self.actual_question_uuid = uuid
        # We dont need to set the question in the context
        # This makes the original question to be added to the context questions
        # self._set_question(question, uuid)

    def _set_question(self, question: str, uuid: str):
        self.context_questions[uuid] = question

    def add_future_questions(self, questions: List[str]):
        for question in questions:
            self.future_questions[uuid4().hex] = question

    def add_context_questions(self, questions: List[str]):
        for question in questions:
            self.context_questions[uuid4().hex] = question

    def get_questions(self) -> List[Tuple[str, str]]:
        """Returns  context questions if they exist, otherwise returns the original question."""
        if len(self.context_questions) > 0:
            return list(self.context_questions.items())
        return [(self.original_question_uuid, self.original_question)]

    async def add_answer(
        self,
        answer: str,
        module: str,
        agent_path: str,
        citations: Optional[AnswerCitations] = None,
        visualization: Visualization | None = None,
        chunks: Optional[list[Chunk]] = None,
        structured: Optional[list[str]] = None,
        images: Optional[Dict[str, Image]] = None,
        image_urls: Optional[list[str]] = None,
    ):
        answer_obj = Answer(
            answer=answer,
            module=module,
            agent_path=agent_path,
            original_question_uuid=self.original_question_uuid,
            actual_question_uuid=self.actual_question_uuid,
            citations=citations,
            chunks=chunks,
            data_visualizations=[visualization] if visualization else None,
            structured=structured,
            images=images,
            image_urls=image_urls,
        )
        self.answers.append((answer, citations))
        if visualization is not None:
            self.data_visualizations.append(visualization)

        if self.callback_fn is not None:
            answer_obj_to_send = AragAnswer(
                possible_answer=answer_obj,
            )
            await self.callback_fn(answer_obj_to_send)

    async def add_final_answer(self):
        if len(self.answers) == 0:
            logger.info("No answers found")
            return
        else:
            answer, citations = self.answers[-1]
            self.final_answer = answer
            self.final_answer_citations = citations

    async def send_final_answer(self):
        if len(self.answers) == 0:
            logger.info("No answers to send")
            return
        else:
            answer, citations = self.answers[-1]
            self.final_answer = answer
            answer_obj = AragAnswer(
                answer=answer,
                original_question_uuid=self.original_question_uuid,
                answer_citations=citations,
            )
            if self.callback_fn is not None:
                await self.callback_fn(answer_obj)

    def show_intermediate_steps(self):
        pass

    async def send_oauth(self, oauth: OAuthAuthenticateURL) -> None:
        if self.oauth_fn is not None:
            return await self.oauth_fn(oauth)
        return None

    async def send_feedback(self, feedback: Feedback) -> UserToAgentInteraction | None:
        if self.feedback_fn is not None:
            return await self.feedback_fn(feedback)
        return None

    async def recv_oauth_callback(
        self, question_id: str, oauth_uuid: str
    ) -> str | None:
        """Receive OAuth callback credentials."""
        if self.oauth_callback_fn is not None:
            return await self.oauth_callback_fn(question_id, oauth_uuid)
        return None

    def set_callback_fn(self, callback: Callable[[AragAnswer], Awaitable[None]]):
        self.callback_fn = callback

    async def emit_streaming_chunk(
        self,
        chunk: StreamingChunk | None = None,
        *,
        reasoning: bool = False,
        agent_request: str | None = None,
    ) -> None:
        """Emit any streaming event through the callback."""
        if self.callback_fn is None:
            return

        answer = AragAnswer(
            operation=AnswerOperation.REASONING
            if reasoning
            else AnswerOperation.ANSWER_CHUNK,
        )
        if agent_request is not None:
            answer.agent_request = agent_request
        if chunk is not None:
            if reasoning:
                answer.reasoning = chunk
            else:
                answer.streaming_response_chunk = chunk
        await self.callback_fn(answer)

    def set_oauth_fn(self, oauth: Callable[[OAuthAuthenticateURL], Awaitable[None]]):
        self.oauth_fn = oauth

    def set_oauth_callback_fn(
        self,
        oauth_callback: Callable[[str, str], Awaitable[str | None]],
    ):
        self.oauth_callback_fn = oauth_callback

    def set_feedback_fn(
        self, feedback: Callable[[Feedback], Awaitable[UserToAgentInteraction | None]]
    ):
        self.feedback_fn = feedback

    async def save(self):
        await self.session.save(self)

    def get_tracking_info(self) -> TrackingInfo:
        """Returns tracking context for propagation to manager calls."""
        return TrackingInfo(
            rao_id=self.get_agent_id(),
            session=self.get_tracking_session(),
            message=self.original_question_uuid,
        )

    def get_tracking_session(self) -> str:
        """Returns a composite session key combining workflow and session IDs for activity tracking."""
        return f"{self.get_workflow_id()}_{self.get_session_id()}"

    def get_streaming(self) -> bool:
        """Returns whether streaming is enabled for this interaction."""
        return self.streaming
