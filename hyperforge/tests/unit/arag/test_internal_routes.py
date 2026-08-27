from hyperforge.api.app import HTTPApplication
from hyperforge.api.settings import Settings
from hyperforge.db.settings import DataManagerSettings


def test_internal_inspection_route_is_not_mounted():
    app = HTTPApplication(
        Settings(),
        DataManagerSettings(postgresql_dsn="postgresql://unused"),
    )

    assert all(
        getattr(route, "path", None) != "/api/internal/v1/agent/{kbid}"
        for route in app.routes
    )
