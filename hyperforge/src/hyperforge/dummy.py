from typing import Any

from stashify_protos.protos import idp_pb2


class DummyIDPRegionalGRPCUtility:
    requests: list[Any]

    def __init__(self) -> None:
        self.requests = []

    async def SendRAOExportEmail(
        self, payload: idp_pb2.SendRAOExportEmailRequest
    ) -> idp_pb2.SendRAOExportEmailResponse:
        self.requests.append(payload)
        return idp_pb2.SendRAOExportEmailResponse(
            status=idp_pb2.SendRAOExportEmailResponse.Status.OK,
            message="Dummy IDP Regional GRPC Utility: Email sent successfully",
        )

    async def SendEmail(
        self, payload: idp_pb2.SendEmailRequest
    ) -> idp_pb2.SendEmailResponse:
        self.requests.append(payload)
        return idp_pb2.SendEmailResponse(
            status=idp_pb2.SendEmailResponse.Status.OK,
            message="Dummy IDP Regional GRPC Utility: Email sent successfully",
        )
