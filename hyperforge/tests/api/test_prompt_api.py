import pytest
from httpx import AsyncClient
from nucliadb_models.resource import KnowledgeBoxObj

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test"]),
    pytest.mark.asyncio,
]

HEADERS = {
    "X-STF-USER": "user1",
    "X-STF-ACCOUNT": "nuclia",
    "X-STF-ACCOUNT-TYPE": "basic",
    "X-STF-ROLES": "SOWNER",
}


async def test_prompt_crud(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    # Create Prompt (using MCP Prompt schema)
    # MCP Prompt has arguments as list of objects
    prompt_payload = {
        "name": "test_prompt",
        "description": "A test prompt",
        "prompt": "In order to do this you should use the tool tool1 with the argument: {arg1} ",
        "arguments": [{"name": "arg1", "description": "Argument 1", "required": True}],
    }

    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/prompts",
        json=prompt_payload,
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    prompt_id = data["id"]

    # List Prompts
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/prompts",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    prompts = resp.json()
    assert len(prompts) > 0
    # Note: get_prompts returns PromptConfig list, so structure might differ from input
    found = False
    for p in prompts:
        # PromptConfig has 'name', 'description', 'text', 'arguments' (dict)
        if p.get("name") == "test_prompt":
            found = True
            break
    assert found

    # Get Prompt
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/prompt/{prompt_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    prompt_data = resp.json()
    assert prompt_data["name"] == "test_prompt"

    # Update Prompt (using PromptConfig schema)
    # PromptConfig: name, description, text, arguments (dict)
    # The API for PATCH uses PromptConfig
    update_payload = {
        "name": "test_prompt_updated",
        "description": "Updated description",
        "prompt": "Updated text content",
        "arguments": [{"name": "arg1", "description": "Argument 1", "required": False}],
    }

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/prompt/{prompt_id}",
        json=update_payload,
        headers=HEADERS,
    )
    assert resp.status_code == 200

    # Verify Update
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/prompt/{prompt_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    updated_data = resp.json()
    assert updated_data["name"] == "test_prompt_updated"
    assert updated_data["description"] == "Updated description"

    # Delete Prompt
    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/prompt/{prompt_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    # Verify Deletion
    # Expecting 404 or 500? Use 404 typically, or verify it's gone from list
    # The get_prompt might raise error or return null.
    # Let's check the list.
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/prompts",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    prompts = resp.json()
    found = False
    for p in prompts:
        if (
            p.get("id") == prompt_id
        ):  # PromptConfig usually doesn't have ID in response unless added?
            # get_prompts returns PromptConfig which doesn't seem to have ID in the model definition in models.py
            # But maybe the backend injects it?
            # Or we match by name, but name changed.
            pass

    # Actually, let's just make sure the updated name is gone if we assume unique names or checking count
    matching = [p for p in prompts if p.get("name") == "test_prompt_updated"]
    assert len(matching) == 0
