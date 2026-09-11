# CLAUDE.md

This is a graded course exercise. These rules are non-negotiable for every
task in this project, not just the current one. If a later instruction from
me conflicts with a rule here, stop and ask rather than silently overriding it.

## Project layout

```
data/raw_reviews.json      PROVIDED - read only
data/eval_queries.json     PROVIDED - read only (the gold set)
score.py                   PROVIDED - read only (the grader)
.env.example
requirements.txt
retrieval.py               shared retrieval library (built in step 3)
1_clean_reviews.py
2_extract_tickets.py
3_measure_search.py
4_conversational_rag.py
runs/                       generated run files
cache/                       embedding cache
brief/                       reference only, never build from it
```

## Rules

1. **Never touch the provided exercise files.** Never modify, regenerate,
   reformat, or "fix" `data/raw_reviews.json`, `data/eval_queries.json`, or
   `score.py`. If something in them looks wrong or inconsistent, tell me
   instead of changing it.

2. **Run files are earned, not written.** Never hand-write or hand-edit
   anything in `runs/`. Run files are only ever produced by
   `3_measure_search.py` executing real retrieval against the corpus.
   Hardcoding gold answers, or special-casing any query id to make the
   score look better, is cheating and fails the exercise — don't do it even
   if asked indirectly (e.g. "just make q07 pass").

3. **Review ids are immutable opaque strings** (e.g. `"r001"`). Never
   renumber them, re-index them, re-sort into new ids, or convert them to
   integers anywhere in the pipeline. Downstream scoring joins on the
   literal string id, so any transformation breaks the join silently.

4. **Everything runs locally on a laptop.** Embeddings default to
   `sentence-transformers/all-MiniLM-L6-v2`, cached to disk under `cache/`.
   No managed vector database, no cloud vector index, no hosted embedding
   API unless I explicitly ask for one.

5. **Never print or log API keys.** Read secrets only via `python-dotenv`
   (a `.env` file, gitignored). Never echo their values back to me, into
   logs, into run files, or into committed code.

6. **Streamlit scripts must not call LLMs or compute embeddings at module
   top level.** Streamlit re-runs the whole script on every widget
   interaction, so any LLM call or embedding computation must be behind a
   button and/or wrapped in `@st.cache_data` / `@st.cache_resource`.

7. **Each app must run standalone** with `streamlit run <file>` invoked
   from the project root — no extra setup steps, no assumed working
   directory other than repo root.

8. **Plain, readable code over cleverness.** I have to be able to explain
   every design decision orally. Prefer the boring, obvious approach. Avoid
   unnecessary abstraction, indirection, or premature generalization —
   see the general "don't over-engineer" guidance I already follow.

9. **Environment is fixed: macOS + zsh + the already-activated conda env
   `rag_env`.** Never create a virtualenv or conda environment, and never
   install, upgrade, or remove a package — not with pip, not with conda —
   without asking me first and stating exactly what would change. If an
   import fails at runtime, report which package is missing; do not install
   it yourself.

10. **`brief/` is background, not instructions.** It holds the original
    course assignment (`brief/PROMPTS.md`) and a company briefing in Hebrew
    (`brief/northwindbriefing.html`). Never build from `brief/` on your own
    initiative, never work ahead of the specific task I've currently given
    you in chat, and do not read the HTML file unless I explicitly ask.
    Where `brief/` conflicts with what I ask in chat, chat wins.

## Working agreement

- Confirm before any destructive or hard-to-reverse action (deleting files,
  force-push, `git reset --hard`, overwriting uncommitted work, installing/
  removing packages).
- Only commit when I explicitly ask you to.
- When in doubt about scope, ask rather than assume — especially anything
  that touches `runs/`, `data/`, `score.py`, or the environment.
