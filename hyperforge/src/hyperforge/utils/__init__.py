import ipaddress
import socket
import urllib.parse
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import aiodns
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


def sync_check_dns(url: str) -> Optional[str]:
    parsed_url = urllib.parse.urlparse(url)
    error = None
    try:
        if parsed_url.hostname is None:
            hostname = parsed_url.path
        else:
            hostname = parsed_url.hostname
        addr = socket.gethostbyname_ex(hostname)
        for addres in addr[2]:
            if ipaddress.ip_address(addres).is_private:
                error = "Its a private address"
    except aiodns.error.DNSError:
        error = "Could not find this URL"
    return error


def sync_dns_validation(url: str) -> str:
    error = sync_check_dns(url)
    if error is not None:
        raise ValueError(error)
    return url


async def check_dns(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    resolver = aiodns.DNSResolver()
    error = None
    try:
        if parsed_url.hostname is None:
            hostname = parsed_url.path
        else:
            hostname = parsed_url.hostname
        addr = await resolver.gethostbyname(hostname, socket.AF_INET)
        for addres in addr.addresses:
            if ipaddress.ip_address(addres).is_private:
                error = "Its a private address"
    except aiodns.error.DNSError:
        error = "Could not find this URL"

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
    """Selector for data source drivers (databases, APIs, etc.)"""

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
