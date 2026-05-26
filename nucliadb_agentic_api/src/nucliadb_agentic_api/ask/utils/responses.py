from fastapi.responses import JSONResponse


class HTTPClientError(JSONResponse):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, content={"detail": detail})
