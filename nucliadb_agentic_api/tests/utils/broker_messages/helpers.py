from nucliadb_protos.resources_pb2 import Classification


def labels_to_classifications(
    labelset: str, labels: list[str], split: str | None = None
) -> list[Classification]:
    classifications = []
    for label in labels:
        classification = Classification(
            labelset=labelset,
            label=label,
            cancelled_by_user=False,
        )
        if split is not None:
            classification.split = split
        classifications.append(classification)
    return classifications
