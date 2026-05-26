import os
import pathlib

# from base64 import b64encode
import pytest
from hyperforge.configure import GLOBAL_REGISTRY
from hyperforge.engine import init
from hyperforge.interaction import AragAnswer, Feedback
from hyperforge.memory.memory import (
    EphemeralSessionMemory,
    QuestionMemory,
)
from hyperforge.pubsub import UserToAgentInteraction
from tests.arag import NUA_KEY
from tests.arag.psco_questions import (
    QUESTIONS,
)

_dir = pathlib.Path(__file__).parent.absolute()
_package_path = _dir.parent.absolute()


with open(f"{_package_path}/arag/restricted_python.py", "r") as f:
    raw_code = f.read()
    CODE = raw_code.split("## SPLIT HERE ##")[-1]

CONFIG = {
    "drivers": [
        {
            "provider": "google",
            "identifier": "google-01",
            "name": "google",
            "config": {
                "vertexai": False,
                "api_key": "AIzaSyDBBq0QwyVtYauiP0D7GkqQaWm8A92kHrM",
            },
        },
        {
            "name": "pepsico-sec",
            "provider": "nucliadb",
            "identifier": "pepsico-sec",
            "config": {
                "url": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "manager": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "kbid": "aad544f3-9131-4c70-ba55-5937d31d53d5",
                "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2F3cy11cy1lYXN0LTItMS5udWNsaWEuY2xvdWQvIiwiaWF0IjoxNzYzNjM5ODc1LCJzdWIiOiI5YTg2ZmMyYS04OWNmLTQxMTktYTY1ZC02ODM5M2NmYTZlNjkiLCJqdGkiOiJhYmYyZWJmZi0xMGJlLTQyNGQtOWQ4Yi00ZjkyZGY0ZWQ0NzAiLCJleHAiOjE3OTUxNzU4NzMsImtleSI6ImZhMWE4YmE4LTNkMmItNDg2Yi1iM2EwLTZjNTdlNTY0NDJjZSIsImtpZCI6ImVhYzdkOTJmLTkzZmItNGYxYy1hZTc5LTM3NWY0NjYyYmJlMSJ9.WRhAvrBAzMX1rsGPkoIV-YdMqvl0_SGxErUBeVvZi9wpV3Lq5qhENuswuZPqroCjHPYz9u1QU_rd-td-PI5l-p0tucA2UUTBzP7VVNstHzfWlFaA7JGqixjtW6ex4dESflSkRRhM98mvSAoVnXlVSqZr0BfmLI1jzYuSG5nEuL07ovJWZjmQ6xAdyL64QT0ESoiiuaE0nGqHom5DvfeoopoTOUjugBT16trVyB5mW2h8DQzhJKp32mpuh4uLcSFL5lOVQzq75rWTG0K4q-clyHFrAibgo5OwtT-Wz1WwvJQJf8tu9QOk-wRLKDMOssNw14e6KH1SeuYV0N07ChOHRVeFO5fOdmEuNcxdEQJvuL0_8FQ0U6dRZ6yJBAMWTg5hvQcKd5SIuHghdLzNP0lnOXnINHx-ZaHUgoDKbpbHTzBYOjlFak4DOeM4Dl0Y2wLXnbEXuw5U5-XyQwMxAVdAk--waUSr0A_G0numvQ7JG-yOKn5bAYKm8mOeqwGC0FxPOBYBQY-U1nGK8Yn_SL-xqs4B8a7xKQxm32_hIHBQ8v8kRzO1r08DAk50nQ6EBrh9tuswdL2pppJm-ws2-gQxQ11KMzF0DZck_yrumQcpIabKvq2D3ePbZaIzBEXZkWtIAI4PdoBz7X4PowfSW1ptOijgV-fgfNUHlsEdPZGkXzs",
                "filters": [],
                "description": "SEC Documents Database",
            },
        },
        {
            "name": "pepsico-sc",
            "provider": "nucliadb",
            "identifier": "pepsico-sc",
            "config": {
                "url": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "manager": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "kbid": "83070996-146e-4a51-9701-1afad6e35fee",
                "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2F3cy11cy1lYXN0LTItMS5udWNsaWEuY2xvdWQvIiwiaWF0IjoxNzYzNjM5OTc0LCJzdWIiOiI4NDllNDY3ZC1lMTA5LTRmYzctYjVjYi0wOGRhNTRiYTk4ZDciLCJqdGkiOiJlNjJiZDYyMi1iY2U1LTRkM2EtODFjZC1iNmY4MGFjYzM0NTgiLCJleHAiOjE3OTUxNzU5NzEsImtleSI6ImM4NTEyNGZjLTVlMmMtNDQzNy1iNGRkLTdlNzAzNDBlMGYxZCIsImtpZCI6ImE4NDQ0Mzg5LTYwNzItNDEwYi05YWY5LTU2OTQ1ZGYzMjFkMCJ9.DsxbkUgMgk7-7c3UTUOrA_JgD62cYYhyRcjZtuVih6jik_1MRglCAQYl35-Xiutj2g2nNPo4r_UJXWZSMuukU0JGTZYygAVDjuPhbwF6WGEzge2BIUplMyZa0y9IMDzymYBZlAKdAplgPD-wBWnovWGogOo8I17M9JuRHQYH3kl_vNMgREpyV9k-zFoeXchDroCE2B6L9IPBCaFl2fO0o7Lcjv7ebwg8fXEaBCQGk4cMCcgwCBn8oIxZuYn1JQw85_3qBC05EH_nSBVU2cow_G9QcWrulFa_f4hn3aU1-q4Ikyjb4A0JYjfr9B-AfJFMwKVJldNQWa53BrLbm6kT3aUT1WMHMi-nHZiAh7_r_uC0zb70ZmkN5MVJk0sMCvv2rjLr6CWmhpJEOphOJcIobxtoUOIOhuyrZ8lSbMbj9mD_PHEJDRXPr6jUUMOqmRAwaXe4jGO7C4irFOt8k0araip0W2DaHnrbevNSA5ftxjsJArZN6wB6fbPBFkuF7achQleqXsvpwvIXStapTKISTZc4N76z2hOh1nJJStEdw1GOYOPBUPCtPxVa5ZDqRgeGiXDCoPnw6TaDrmpwwQ9Sqy1M4A5_aY-qUEWAojYkXUUxGmhtD6ZqxouTC_bplTSXf2f_1Jou12paBuu9jBt_00Fgu5UfjZFpJAg-043owyg",
                "filters": [],
                "description": "SC Documents Database",
            },
        },
        {
            "name": "pepsico-physics",
            "provider": "nucliadb",
            "identifier": "pepsico-physics",
            "config": {
                "url": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "manager": "https://aws-us-east-2-1.rag.progress.cloud/api",
                "kbid": "5d30e71d-58a6-4d3e-ab35-b77d404ab631",
                "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2F3cy11cy1lYXN0LTItMS5udWNsaWEuY2xvdWQvIiwiaWF0IjoxNzYzNjQwMTIyLCJzdWIiOiJhYjNiMzcyYi0xYzIxLTQwY2YtOGVkZC03Mjk4ZDU2OTdhYzAiLCJqdGkiOiI0N2EwNDMxYy04NTEwLTQ1OTctYWIwMC0yN2EwYjVlOGMzZjAiLCJleHAiOjE3OTUxNzYxMjAsImtleSI6ImYzMjc1Y2Y4LTMzZTUtNGJjMi1hNTAyLWZmMzI5NTE0MmE2NCIsImtpZCI6ImRiMDNhNjA5LWFiOTAtNDFkMi05MDQ3LTcxNGUwOWQxYWE2NCJ9.jCvMWUjInkdFYS1hB10SK5skQ_g7pliwLnPVS3_lRA8_SoVqLLSDuXT8108XC2TT4jRbzn5uqJDXY68QDIQm3ATvs0QicbLlRH8bR0kP6LgGVZbXMIW5iEvitJ66ijvFni_TBexcSpqoDkJFkZQNNMI1blL5nlBTo53IfkkvILIl6eeLK0ic5H_zaUXkb9q975EURn0CQX3YDs1VTku83oAVIBv1Y1e6gptkfleVwGJOlnFWXOB7NMCWDN87TjuMd5dmag7HhmnX8Iggc5zYOl1s5kagF00BDkrt-6h2dvnys2Qx24qlCPMmv9UUxWLYHASWA0i5Dm-ODEiMx4Fup_NyEC7YyW0DPOTe7jzP2m05ZOwK0nB1v23EFvB5ibhPTu6nv_Ja1Bm2JuES9YWgljsIlCgSF3lXeXXudY0MmOUwsryritYJbRJPGMhIyFILXNJgssmU7e3i7bC8rOfLxaVJ7SlxHS1s6FyXvtc9ToJqtOmDMKhk0fNaJGBJ8TLj0OTA_NT9yUnLc_AqYXNt3PX3hqpXDIgG7HnPXT7v1CtkkBFNZ6eoSsNhLXvBuoLF1XkS1O3PR10Wn0nNn45QYAfp-4CvC0Qp0EgiUjJGmkmiBitACTPHG3TMZtb8hIlhKJWeJczDLwk30xhEKggv58V9puxma7rXxS9AW4h_G1o",
                "filters": [],
                "description": "Physics Documents Database",
            },
        },
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [],
    "context": [
        {
            "module": "restricted",
            "title": "",
            "agents": [
                {
                    "id": "internet-agent",
                    "module": "google",
                    "gen_model_id": "gemini-3-flash-preview",
                    "title": "",
                    "source": "google-01",
                },
                {
                    "id": "pepsico-sec-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-sec"],
                    "generative_model": "gcp-claude-4-5-haiku",
                },
                {
                    "id": "pepsico-sc-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-sc"],
                    "generative_model": "gcp-claude-4-5-haiku",
                },
                {
                    "id": "pepsico-physics-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-physics"],
                    "generative_model": "gcp-claude-4-5-haiku",
                },
            ],
            "decision_model": "gcp-claude-4-5-haiku",
            "code": CODE,
            "parameters": {"internal": "bool", "dataset": "str", "internet": "bool"},
            "debug": True,
            "max_retries": 10,
            "needs_rephrase": True,
        },
    ],
    "generation": [
        {
            "module": "advanced_generation",
            "summarize_config": {
                "module": "summarize",
                "model": "gcp-claude-4-5-haiku",
                "citations": True,
            },
            "data_viz_enabled": True,
            "data_viz_config": {
                "module": "data_viz",
                "model": "gcp-claude-4-5-sonnet",
                "title": "Data Visualization Generator",
            },
        }
    ],
    "postprocess": [],
}


async def callback(arag_answer: AragAnswer):
    print(65 * "_")
    print(arag_answer)
    print(65 * "_")


async def feedback(feedback_input: Feedback) -> UserToAgentInteraction | None:
    print("Feedback received:", feedback_input)
    return None


@pytest.mark.skipif(
    os.environ.get("LOCAL_TESTING") is None,
    reason="Only check if LOCAL_TESTING var is enabled",
)
async def test_restricted_agent():
    state, memory = await init(
        config=CONFIG,
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        memory_klass=EphemeralSessionMemory,
    )
    # memory.
    for question in QUESTIONS:
        preanswer = question.get("preanswer", False)
        internal = question["internal"]
        custom_dataset = question.get("dataset", None)
        internet = question["internet"]
        question_text = question["question"]
        origin_urls = question.get("origin_urls", False)
        if preanswer is False:
            question_memory = memory.start_question(question_text)
            question_memory.set_callback_fn(callback)
            question_memory.set_feedback_fn(feedback)
            question_memory.headers["X-RESTRICTED-INTERNAL"] = str(internal)
            question_memory.headers["X-RESTRICTED-DATASET"] = str(custom_dataset)
            question_memory.headers["X-RESTRICTED-INTERNET"] = str(internet)
            await state.agent(question_memory, state.manager)
            print(question_memory.final_answer)
            result, _, _ = await state.manager.execute_json(
                prompt=f"Are the two answers to the question {question_text} the same? Answer1: {question_memory.final_answer} Answer2: {question['answer']} Answer yes or no.",
                user_id="test-user",
                model="gcp-claude-4-5-haiku",
                schema={
                    "type": "object",
                    "title": "Answer Comparison",
                    "description": "Compare two answers to determine if they are the same.",
                    "properties": {
                        "same_answer": {"type": "string", "enum": ["yes", "no"]}
                    },
                },
            )
            assert result["same_answer"] == "yes", "Answers do not match"

            assert question_memory.is_answered, "Question not answered"

            assert question_memory.final_answer_citations, "No citations found"
            if origin_urls:
                # Check that there are origin urls collected in the citations metadata
                assert any(
                    len(citation.origin_urls) > 0
                    for citation in question_memory.final_answer_citations.metadata.values()
                )

            if question.get("generated_chart", False):
                assert question_memory.contexts[-1].images, "No images generated"
            await question_memory.save()
        else:
            fake_qm = QuestionMemory(
                memory, question_text, actions=None, question_id=None, headers={}
            )
            fake_qm.final_answer = question["answer"]
            await fake_qm.save()
    GLOBAL_REGISTRY.clear()
