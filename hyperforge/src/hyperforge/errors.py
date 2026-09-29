import asyncio
import re
from collections.abc import Iterator

_AUTH_CREDENTIAL = re.compile(r"(?i)\b(Bearer|Basic)\s+[^\s,;]+")
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)\b(authorization)(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|(?:(?:Basic|Bearer)\s+)?[^\s,;]+)"
)
_COOKIE_VALUE = re.compile(r"(?i)\b(set-cookie|cookie)(\s*:\s*)[^\r\n]+")
_SECRET_KEYS = (
    r"api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"credential|password|private[_-]?key|refresh[_-]?token|secret|session[_-]?id|"
    r"oauth[_-]?token|token"
)
_QUOTED_SECRET_VALUE = re.compile(
    rf"(?i)\b({_SECRET_KEYS})\b(\s*[\"']?\s*[:=]\s*)([\"'])[^\"']*\3"
)
_SECRET_VALUE = re.compile(
    rf"(?i)\b({_SECRET_KEYS})\b(\s*[\"']?\s*[:=]\s*)([^\"',\s;&}}]+)"
)
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^\s/:@]+:[^\s/@]+@")


def _redact_secrets(detail: str) -> str:
    detail = _URL_CREDENTIALS.sub(r"\1[REDACTED]:[REDACTED]@", detail)
    detail = _AUTHORIZATION_VALUE.sub(r"\1\2[REDACTED]", detail)
    detail = _AUTH_CREDENTIAL.sub(r"\1 [REDACTED]", detail)
    detail = _COOKIE_VALUE.sub(r"\1\2[REDACTED]", detail)
    detail = _QUOTED_SECRET_VALUE.sub(r"\1\2\3[REDACTED]\3", detail)
    return _SECRET_VALUE.sub(r"\1\2[REDACTED]", detail)


def _leaf_exceptions(exc: BaseException) -> Iterator[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _leaf_exceptions(nested)
        return

    yield exc


def actionable_exception_group(exc: BaseExceptionGroup) -> BaseExceptionGroup:
    """Return regular exceptions while allowing only cancellation to be discarded."""
    actionable, remainder = exc.split(Exception)
    if actionable is None or (
        remainder is not None
        and any(
            not isinstance(item, asyncio.CancelledError)
            for item in _leaf_exceptions(remainder)
        )
    ):
        raise exc
    return actionable


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status

    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def _exception_detail(exc: BaseException) -> str:
    detail = _redact_secrets(str(exc).strip() or type(exc).__name__)
    status = _http_status(exc)
    if status == 401:
        return f"Authentication failed (HTTP 401): {detail}"
    if status == 403:
        return f"Authorization failed (HTTP 403): {detail}"
    return detail


def _exception_priority(exc: BaseException) -> int:
    status = _http_status(exc)
    if status == 401:
        return 0
    if status == 403:
        return 1
    return 2


def exception_detail(exc: BaseException) -> str:
    """Return useful, bounded details from an exception or exception group."""
    leaves = list(_leaf_exceptions(exc))
    actionable = [
        item for item in leaves if not isinstance(item, asyncio.CancelledError)
    ]
    if actionable:
        leaves = actionable

    leaves.sort(key=_exception_priority)
    details = list(dict.fromkeys(_exception_detail(item) for item in leaves))
    if len(details) == 1:
        return details[0]

    visible_details = details[:3]
    if len(details) > len(visible_details):
        visible_details.append(
            f"and {len(details) - len(visible_details)} more error(s)"
        )
    return "Multiple errors: " + "; ".join(visible_details)
