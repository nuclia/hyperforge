from dataclasses import dataclass

from nucliadb_models.common import FieldTypeName
from nucliadb_protos.resources_pb2 import FieldType

FIELD_TYPE_STR_TO_PB: dict[str, FieldType.ValueType] = {
    "t": FieldType.TEXT,
    "f": FieldType.FILE,
    "u": FieldType.LINK,
    "a": FieldType.GENERIC,
    "c": FieldType.CONVERSATION,
}

FIELD_TYPE_PB_TO_STR = {v: k for k, v in FIELD_TYPE_STR_TO_PB.items()}

FIELD_TYPE_NAME_TO_STR = {
    FieldTypeName.TEXT: "t",
    FieldTypeName.FILE: "f",
    FieldTypeName.LINK: "u",
    FieldTypeName.GENERIC: "a",
    FieldTypeName.CONVERSATION: "c",
}

FIELD_TYPE_STR_TO_NAME = {v: k for k, v in FIELD_TYPE_NAME_TO_STR.items()}


@dataclass
class FieldId:
    """
    Field ids are used to identify fields in resources. They usually have the following format:

        `rid/field_type/field_key`

    where field type is one of: `t`, `f`, `u`, `a`, `c` (text, file, link, generic, conversation)
    and field_key is an identifier for that field type on the resource, usually chosen by the user.

    In some cases, fields can have subfields, for example, in conversations, where each part of the
    conversation is a subfield. In those cases, the id has the following format:

        `rid/field_type/field_key/subfield_id`

    Examples:

    >>> FieldId(rid="rid", type="u", key="my-link")
    FieldID("rid/u/my-link")
    >>> FieldId.from_string("rid/u/my-link")
    FieldID("rid/u/my-link")
    """

    rid: str
    type: str
    key: str
    # also knwon as `split`, this indicates a part of a field in, for example, conversations
    subfield_id: str | None = None

    @classmethod
    def from_string(cls, value: str) -> "FieldId":
        """
        Parse a FieldId from a string
        Example:
        >>> fid = FieldId.from_string("rid/u/foo")
        >>> fid
        FieldId("rid/u/foo")
        >>> fid.type
        'u'
        >>> fid.key
        'foo'
        >>> FieldId.from_string("rid/u/foo/subfield_id").subfield_id
        'subfield_id'
        """
        parts = value.split("/")
        if len(parts) == 3:
            rid, _type, key = parts
            _type = cls._parse_field_type(_type)
            return cls(rid=rid, type=_type, key=key)
        elif len(parts) == 4:
            rid, _type, key, subfield_id = parts
            _type = cls._parse_field_type(_type)
            return cls(
                rid=rid,
                type=_type,
                key=key,
                subfield_id=subfield_id,
            )
        else:
            raise ValueError(f"Invalid FieldId: {value}")

    @classmethod
    def from_pb(
        cls,
        rid: str,
        field_type: FieldType.ValueType,
        key: str,
        subfield_id: str | None = None,
    ) -> "FieldId":
        return cls(
            rid=rid,
            type=FIELD_TYPE_PB_TO_STR[field_type],
            key=key,
            subfield_id=subfield_id,
        )

    @property
    def pb_type(self) -> FieldType.ValueType:
        return FIELD_TYPE_STR_TO_PB[self.type]

    @property
    def type_name(self) -> FieldTypeName:
        return FIELD_TYPE_STR_TO_NAME[self.type]

    def full(self) -> str:
        if self.subfield_id is None:
            return f"{self.rid}/{self.type}/{self.key}"
        else:
            return f"{self.rid}/{self.type}/{self.key}/{self.subfield_id}"

    def full_without_subfield(self) -> str:
        return f"{self.rid}/{self.type}/{self.key}"

    def short_without_subfield(self) -> str:
        return f"/{self.type}/{self.key}"

    def paragraph_id(self, paragraph_start: int, paragraph_end: int) -> "ParagraphId":
        """Generate a ParagraphId from the current field given its start and
        end.

        """
        return ParagraphId(
            field_id=self,
            paragraph_start=paragraph_start,
            paragraph_end=paragraph_end,
        )

    def __str__(self) -> str:
        return self.full()

    def __repr__(self) -> str:
        return f"FieldId({self.full()})"

    def __hash__(self) -> int:
        return hash(self.full())

    @staticmethod
    def _parse_field_type(_type: str) -> str:
        if _type not in FIELD_TYPE_STR_TO_PB:
            # Try to parse the enum value
            # XXX: This is to support field types that are integer values of FieldType
            # Which is how legacy processor relations reported the paragraph_id
            try:
                type_pb = FieldType.ValueType(int(_type))
            except ValueError:
                raise ValueError(f"Invalid FieldId: {_type}")
            if type_pb in FIELD_TYPE_PB_TO_STR:
                return FIELD_TYPE_PB_TO_STR[type_pb]
            else:
                raise ValueError(f"Invalid FieldId: {_type}")
        return _type


@dataclass
class ParagraphId:
    field_id: FieldId
    paragraph_start: int
    paragraph_end: int

    @classmethod
    def from_string(cls, value: str) -> "ParagraphId":
        parts = value.split("/")
        paragraph_range = parts[-1]
        start, end = map(int, paragraph_range.split("-"))
        field_id = FieldId.from_string("/".join(parts[:-1]))
        return cls(field_id=field_id, paragraph_start=start, paragraph_end=end)

    @property
    def rid(self) -> str:
        return self.field_id.rid

    def full(self) -> str:
        return f"{self.field_id.full()}/{self.paragraph_start}-{self.paragraph_end}"

    def __str__(self) -> str:
        return self.full()

    def __repr__(self) -> str:
        return f"ParagraphId({self.full()})"

    def __hash__(self) -> int:
        return hash(self.full())
