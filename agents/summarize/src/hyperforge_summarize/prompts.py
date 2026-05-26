MARKDOWN_TWO_LEVELS_CITATIONS_PROMPT_ADJUSTMENT = """
You are given source blocks with IDs like: block-AB, block-BA, block-CD, etc. or block-AB-1, block-BA-2, block-CD-3, etc.
When producing an answer, cite these sources precisely using markdown footnotes.

CITATION RULES

1. In the main body, cite sources only with bracketed Arabic numerals: [1], [2], [3], etc.
  - Never put a block ID directly in brackets (e.g. NO: [AB], [block-AB], [BA]).
  - Never mix styles (no superscripts, no inline block names, no [Ref 1], etc.).
2. Numbering is assigned in order of FIRST USE of a unique block ID.
  - The first time you need info from a block, assign it [1].
  - The first time you need info from a never-before-used block, assign it the next unused number (e.g. [2]).
  - If you later cite the SAME block again, REUSE its existing number (do NOT create a new one).
  - This guarantees there are no duplicate footnote definitions and no gaps.
3. If facts in a sentence come from different blocks, you may concatenate citations WITH spaces: like [1] [3]. Do NOT merge them (no ranges like [1-3]) or output them without spaces (no [1][3]).
4. At the end, output section consisting ONLY of the unique citation mappings, one per line, in ascending numeric order
  - Don't title this section or add any extra text, just the mappings.
  - No duplicates.
  - No skipped numbers.
  - ONLY include blocks actually cited in the body.
5. Do NOT hallucinate block IDs. Only use those provided in the context.

FORMATTING CONTRACT

* Body: free text with numeric citations as specified.
* A blank line.
* References section (if any) exactly as described, no heading.


Example format:

---
"The OP-1 has a built-in tape feature with 6 minutes of recording time [1] [2]. You can record to any of the 4 individual tracks [1]."

[1]: block-AB
[2]: block-FZ-2
---

"""
