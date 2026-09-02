from hyperforge import logger
from hyperforge.manager import Manager
from hyperforge.models import Context
from nuclia.lib.nua_responses import RerankModel


async def rerank(context: Context, manager: Manager, top_k: int) -> None:
    """
    Rerank the context chunks based on the analysis.
    """
    if not context.chunks:
        return

    re_ranked = await manager.rerank(
        RerankModel(
            context={chunk.chunk_id: chunk.text for chunk in context.chunks},
            question=context.question,
            user_id="arag-ask-rerank",
        )
    )

    # Sort chunks by score
    re_ranked_sorted = sorted(
        re_ranked.context_scores.items(), key=lambda x: x[1], reverse=True
    )

    re_ranked_sorted = list(
        filter(
            lambda x: x[1] > 0.05,
            re_ranked.context_scores.items(),
        )
    )

    re_ranked_sorted = re_ranked_sorted[:top_k]

    # If the analysis has a rerank function, use it to rerank the chunks
    old_chunks = {chunk.chunk_id: chunk for chunk in context.chunks}
    context.chunks = []
    for rerank_chunk_id, _ in re_ranked_sorted:
        try:
            chunk = old_chunks[rerank_chunk_id]
            context.chunks.append(chunk)
        except KeyError:
            logger.error(f"Chunk {rerank_chunk_id} not found in old_chunks")
