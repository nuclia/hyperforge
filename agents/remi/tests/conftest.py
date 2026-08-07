import json

pytest_plugins = ["hyperforge.minimal_fixtures"]


def nua_chat_match(request, recorded_request):
    if request.path != "/api/v1/predict/chat":
        return request.body == recorded_request.body
    if recorded_request.path != "/api/v1/predict/chat":
        return False

    request_payload = json.loads(request.body)
    recorded_payload = json.loads(recorded_request.body)
    return all(
        request_payload.get(field) == recorded_payload.get(field)
        for field in ("question", "user_id", "generative_model")
    )


def pytest_recording_configure(config, vcr):
    vcr.register_matcher("nua_chat", nua_chat_match)
