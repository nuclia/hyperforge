import os
import pathlib
from base64 import b64encode

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.memory.memory import (
    EphemeralSessionMemory,
)
from tests.arag import NUA_KEY

_dir = pathlib.Path(__file__).parent.absolute()
_package_path = _dir.parent.absolute()

if os.environ.get("REMOTE_TESTING") is not None:
    with open(f"{_package_path}/arag/assets/SEC Data.xlsx", "rb") as f:
        sec_data = b64encode(f.read()).decode("utf-8")
    with open(
        f"{_package_path}/arag/assets/Computer Science Literature Cleaned.xlsx", "rb"
    ) as f:
        sc_data = b64encode(f.read()).decode("utf-8")
    with open(f"{_package_path}/arag/assets/Physics cleaned new.xlsx", "rb") as f:
        physics_data = b64encode(f.read()).decode("utf-8")
else:
    sec_data = ""
    sc_data = ""
    physics_data = ""


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
            {
                "prompt": "The documentation of Nuclia is hosted at https://docs.nuclia.dev"
            },
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [
        {
            "module": "rephrase",
            "history": True,
            "session_info": True,
            "split_question": True,
            "provided_synonyms": {
                "PepsiCo": ["Pepsi", "Pepsi Company", "Pepsi Corporation"],
                "Coca-Cola": ["Coke", "Coca Cola Company", "Coca Cola Corporation"],
                "Keurig Dr Pepper": ["Keurig", "Dr Pepper", "Keurig and Dr Pepper"],
                "National Beverage Corp.": ["National Beverage", "Nat Bev", "NatBev"],
                "Monster Beverage": ["Monster", "Monster Inc.", "Monster Corporation"],
                "Kellanova": [
                    "Kellanova",
                    "Kellogs",
                ],
            },
        }
    ],
    "context": [
        {
            "module": "smart",
            "title": "",
            "prompt": """Always try to answer the question using the non internet agents first. Use the internet agent only if the non-internet agents do not have the information needed to answer the question.""",
            "registered_agents_descriptions": {
                "internet-agent": "Provides information from the internet. useful for current events and general knowledge.",
                "pepsico-sec-nucliadb": "Provides information about Pepsico and its financials, also about relevant competitors, such as Coca-Cola, Keurig Dr Pepper, National Beverage Corp., Monster Beverage, and Kellanova.",
                "pepsico-sc-nucliadb": "Paper literaure about computer science",
                "pepsico-physics-nucliadb": "Provides information from papers about physics.",
                "pepsico-sec-pandas": "Metadata and relevant information about the files indexed in the Nuclia knowledge box for SEC data. It has information about the names of the files, the company they belong to, the sector, where their headquarters are, and the date of the report.",
                "pepsico-physics-pandas": "Provides relevant metadata and information about the files indexed in the Nuclia knowledge box for Physics data. it has info about the paper titles, authors, universities, countries, keywords, and emails.",
                "pepsico-sc-pandas": "Provides relevant metadata and information about the files indexed in the Nuclia knowledge box for Computer Science literature. It has info about the paper titles, authors, universities, countries, keywords, and emails.",
                "perplexity-agent": "Provides general knowledge and web search capabilities.",
            },
            "registered_agents": [
                {
                    "id": "internet-agent",
                    "module": "google",
                    "title": "",
                    "source": "google-01",
                },
                {
                    "id": "pepsico-sec-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-sec"],
                },
                {
                    "id": "pepsico-sc-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-sc"],
                },
                {
                    "id": "pepsico-physics-nucliadb",
                    "module": "basic_ask",
                    "title": "",
                    "sources": ["pepsico-physics"],
                },
                {
                    "id": "pepsico-sec-pandas",
                    "module": "pandas",
                    "title": "",
                    "xlsx_base64": sec_data,
                },
                {
                    "id": "pepsico-physics-pandas",
                    "module": "pandas",
                    "title": "",
                    "xlsx_base64": physics_data,
                },
                {
                    "id": "pepsico-sc-pandas",
                    "module": "pandas",
                    "title": "",
                    "xlsx_base64": sc_data,
                },
                {
                    "module": "perplexity",
                    "source": "perplexity-01",
                    "title": "Perplexity Agent",
                },
            ],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


@pytest.mark.skipif(
    os.environ.get("LOCAL_TESTING") is None,
    reason="Only check if LOCAL_TESTING var is enabled",
)
@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_smart_demo():
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Which companies compete with PepsiCo in the beverages sector?",
        memory_klass=EphemeralSessionMemory,
        config=CONFIG,
    )
    keywords = ["monster", "coca-cola", "red bull", "nestlé"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
