import urllib.parse
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from nuclia_models.predict.generative_responses import GenerativeFullResponse


def iterate_tools_resp(
    resp: GenerativeFullResponse,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if resp.tools is not None:
        for _, tools_calls in resp.tools.items():
            for tool in tools_calls:
                if tool.function.name:
                    used_params = tool.function.arguments
                    yield tool.function.name, used_params


def validate_url(url: str) -> Optional[str]:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"}:
        return "URL must use HTTP or HTTPS"
    if parsed_url.hostname is None:
        return "URL must include a hostname"
    if parsed_url.username is not None or parsed_url.password is not None:
        return "Credentials are not allowed in URLs"
    return None


def sync_dns_validation(url: str) -> str:
    error = validate_url(url)
    if error is not None:
        raise ValueError(error)
    return url


class WidgetType(str, Enum):
    """
    Enumeration of available widget types for form field rendering.
    These correspond to the frontend component mappings.
    """

    # Model and AI-related widgets
    MODEL_SELECT = "model_select"
    """Dropdown selector for AI models (LLM, embedding, etc.)"""

    # Driver and source selection widgets
    DRIVER_SELECT = "driver_select"
    """Selector for data sources (databases, APIs, etc.)"""

    FILTERED_SOURCE_SELECT = "filtered_source_select"
    """Source selector with filtering capabilities based on transport type"""

    # Code and text input widgets
    CODE_EDITOR = "code_editor"
    """Code editor with syntax highlighting, supports multiple languages"""

    EXPANDABLE_TEXTAREA = "expandable_textarea"
    """Multi-line text input that can expand/resize"""

    # Specialized field widgets
    TRANSPORT_FIELD = "transport_field"
    """Selector for transport/protocol types (HTTP, gRPC, etc.)"""

    RULES_FIELD = "rules_field"
    """Complex rules configuration interface"""

    KEY_VALUE_FIELD = "key_value_field"
    """Key-value pairs input (like environment variables)"""

    # Basic form widgets
    ARRAY_STRING_FIELD = "array_string_field"
    """Input for arrays of string values"""

    ENUM_SELECT = "enum_select"
    """Dropdown for predefined enumeration values"""

    # Extra for RAO

    NOT_SHOWN = "not_show"
    """Field is not shown in the node config UI"""
