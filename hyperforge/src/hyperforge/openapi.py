import argparse
import datetime
import json

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi
from starlette.routing import compile_path

extract_openapi_parser = argparse.ArgumentParser()
extract_openapi_parser.add_argument("openapi_json_path", type=str)
extract_openapi_parser.add_argument("api_version")
extract_openapi_parser.add_argument("commit_id", type=str)


def extract_openapi_command(component_id: str, title: str, router: APIRouter):
    """
    This function assumes that json, api version and commit id are coming from the command line.

    However, what can be wired in here is the title of the API and the API Router
    that provides the endpoints.

    This function assumes that you want to extract things from a router in the form of:
    - /api/v1 or /api/v2
    """
    args = extract_openapi_parser.parse_args()
    openapi_json_path = args.openapi_json_path
    api_version = args.api_version
    commit_id = args.commit_id

    app = FastAPI(title=title, version=f"{api_version}.0.0")

    route_prefix = f"/api/v{api_version}"
    routes = []
    for route in router.routes:
        # check if route starts with prefix and strip it, then add to new router
        if route.path.startswith(route_prefix):  # type: ignore
            route.path = route.path[len(route_prefix) :]  # type: ignore
            route.path_regex, route.path_format, route.param_convertors = compile_path(  # type: ignore
                route.path  # type: ignore
            )
            routes.append(route)

    document = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
        routes=routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

    document["x-metadata"] = {
        component_id: {
            "commit": commit_id,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    }

    json.dump(document, open(openapi_json_path, "w"))
