from nucliadb_models.search import NucliaDBClientType
from nucliadb_protos.audit_pb2 import ClientType


def client_type(obj: NucliaDBClientType) -> ClientType.ValueType:
    return ClientType.Value(obj.name)
