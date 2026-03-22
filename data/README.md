# Data Layout

This project separates private source materials from public processed knowledge files.

- `raw_private/`: local-only PDFs, slides, textbook excerpts, answer keys, and team-shared materials. Do not upload these files to a public repository.
- `processed/`: public-safe text chunks and summaries created by the author for retrieval and evaluation.
- `netflix_did/`: public-safe chunks focused on the Netflix causal inference case or recommendation analysis project.
- `causal_methods/`: public-safe chunks for methods such as DID, IPW, ATT, ATE, and regression adjustment.
- `project_notes/`: author-written project summaries, experimental reasoning notes, and reusable analysis logic.
- `eval_questions/`: benchmark questions used to compare no-RAG and RAG answers.

Recommended workflow:

1. Place source PDFs or slides in `raw_private/` for local processing only.
2. Extract notes or chunked text into `netflix_did/`, `causal_methods/`, or `project_notes/`.
3. Store reusable public-safe artifacts in `processed/` if you want a separate export area.
4. Cite the source category in metadata without redistributing copyrighted or team-private files.

Some original materials are intentionally excluded from the public repository due to copyright or team-sharing restrictions.
