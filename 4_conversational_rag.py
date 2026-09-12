"""Streamlit app — Step 4: conversational RAG chat over clean_reviews.json.

Uses retrieval.py's hybrid_search at the exact winning configuration from
step 3's tuning (config.py — shared with 3_measure_search.py so the
measured system and this shipped system can never drift apart). Never
calls the generation model when retrieval finds nothing. Every answer must
cite review ids, and every citation is checked against what was actually
retrieved.

Run with: streamlit run "4_conversational_rag.py"
"""
import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import config
import retrieval as rt

load_dotenv()

REFUSAL_MESSAGE = "I don't have information about that in the reviews."
CITATION_LOG_PATH = Path("logs/citation_failures.jsonl")

GEMINI_MODEL = "gemini-3.6-flash"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

CITATION_PATTERN = re.compile(r"\[(r\d{3})\]")

GROUNDING_SYSTEM_PROMPT = """You are a customer-support assistant that answers questions using ONLY the customer reviews supplied below. Follow these rules exactly:

1. Never use any knowledge you have from outside these reviews. If the reviews don't contain the answer, say so plainly instead of guessing.
2. Every claim in your answer must be grounded in one of the reviews below. Cite the review id(s) you used in square brackets, e.g. [r011]. Cite every review you draw on, not just the first one.
3. If two or more of the reviews disagree or contradict each other on the same topic, do not pick one side. State plainly that the reviews disagree, and cite both sides, e.g. "Some reviews report 3 hours [r016], while others report 30 hours [r019]."
4. Keep your answer concise.

Reviews:
{context}"""

REWRITE_SYSTEM_PROMPT = """You rewrite a user's latest chat message into a single standalone search query for a review search engine, using the conversation history to resolve pronouns and implicit references (e.g. "it", "that", "the same one").

Reply with ONLY the standalone query text. No quotes, no explanation, no punctuation beyond what belongs in the query itself."""


# --------------------------------------------------------------------------
# Generation provider dispatch — each SDK imported lazily inside its own
# branch, same isolation pattern as 2_extract_tickets.py.
# --------------------------------------------------------------------------

def _generate_gemini(system_instruction, user_prompt, api_key):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text.strip()


def _generate_openai(system_instruction, user_prompt, api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _generate_anthropic(system_instruction, user_prompt, api_key):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_instruction,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_text(system_instruction, user_prompt):
    provider = os.getenv("PROVIDER", "gemini").lower()
    if provider == "gemini":
        return _generate_gemini(system_instruction, user_prompt, os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return _generate_openai(system_instruction, user_prompt, os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        return _generate_anthropic(system_instruction, user_prompt, os.getenv("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unknown PROVIDER: {provider!r} (expected openai, anthropic, or gemini)")


# --------------------------------------------------------------------------
# RAG turn pipeline — pure functions, no Streamlit calls, so acceptance
# tests can call process_turn directly.
# --------------------------------------------------------------------------

def rewrite_query(history, user_message):
    """Step 1: resolve pronouns/references using chat history. Only meant
    to be called when memory is on AND history is non-empty — a first turn
    has nothing to resolve, so callers skip this rather than spend an API
    call on a no-op rewrite."""
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"User: {user_message}")
    transcript = "\n".join(lines)
    return generate_text(REWRITE_SYSTEM_PROMPT, transcript)


def build_context(retrieved, corpus_by_id):
    return "\n".join(f"[{doc_id}] {corpus_by_id[doc_id]['text']}" for doc_id, _ in retrieved)


def generate_answer(question, context):
    """Step 4. Answers the user's actual question (not the rewritten
    standalone query — that's only for retrieval), grounded in context."""
    system_instruction = GROUNDING_SYSTEM_PROMPT.format(context=context)
    return generate_text(system_instruction, question)


def check_citations(answer, retrieved):
    """Step 5: every [rNNN] in the answer must be among the retrieved ids."""
    cited_ids = set(CITATION_PATTERN.findall(answer))
    retrieved_ids = {doc_id for doc_id, _ in retrieved}
    invented = cited_ids - retrieved_ids
    return {"cited_ids": sorted(cited_ids), "invented_ids": sorted(invented), "passed": not invented}


def process_turn(user_message, history, use_memory, corpus):
    """Orchestrates one chat turn. The early `return` on empty retrieval
    makes the generate_answer call structurally unreachable in that
    branch — not a conditional the model could route around."""
    if use_memory and history:
        standalone_query = rewrite_query(history, user_message)
    else:
        standalone_query = user_message

    retrieved = rt.hybrid_search(standalone_query, k=config.TOP_K, corpus=corpus, **config.HYBRID_KWARGS)

    if not retrieved:
        return {
            "answer": REFUSAL_MESSAGE,
            "standalone_query": standalone_query,
            "retrieved": [],
            "context": "",
            "citation_check": None,
        }

    corpus_by_id = {r["id"]: r for r in corpus}
    context = build_context(retrieved, corpus_by_id)
    answer = generate_answer(user_message, context)
    return {
        "answer": answer,
        "standalone_query": standalone_query,
        "retrieved": retrieved,
        "context": context,
        "citation_check": check_citations(answer, retrieved),
    }


def log_citation_failure(user_message, result):
    CITATION_LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "user_message": user_message,
        "standalone_query": result["standalone_query"],
        "answer": result["answer"],
        "retrieved_ids": [doc_id for doc_id, _ in result["retrieved"]],
        "invented_ids": result["citation_check"]["invented_ids"],
    }
    with open(CITATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

@st.cache_data
def load_corpus():
    return rt.load_reviews()


def render_debug(result):
    with st.expander("Details"):
        st.markdown(f"**Standalone query:** {result['standalone_query']!r}")
        if result["retrieved"]:
            st.markdown("**Retrieved:**")
            for doc_id, score in result["retrieved"]:
                st.markdown(f"- `{doc_id}` ({score:.4f})")
        else:
            st.markdown("**Retrieved:** none")
        st.markdown("**Context passed to the model:**")
        st.code(result["context"] or "(none — generation was not called)")
        check = result["citation_check"]
        if check is None:
            st.markdown("**Citation check:** N/A (no generation call was made)")
        elif check["passed"]:
            st.markdown("**Citation check:** PASS")
        else:
            st.markdown(f"**Citation check:** FAIL — invented ids: {check['invented_ids']}")


def main():
    st.set_page_config(page_title="Conversational RAG", layout="wide")
    st.title("Step 4 — Conversational RAG")

    corpus = load_corpus()
    provider = os.getenv("PROVIDER", "gemini")
    st.caption(f"Provider: **{provider}**  ·  {len(corpus)} cleaned reviews  ·  "
               f"hybrid_search(k={config.TOP_K}, identifier_override={config.HYBRID_KWARGS['identifier_override']})")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations_checked" not in st.session_state:
        st.session_state.citations_checked = 0
    if "citations_failed" not in st.session_state:
        st.session_state.citations_failed = 0

    use_memory = st.sidebar.checkbox("Conversation memory", value=True,
                                      help="When on, the chat history is used to resolve "
                                           "pronouns (\"it\", \"that\") into a standalone "
                                           "search query before retrieval.")
    st.sidebar.metric("Citations checked", st.session_state.citations_checked)
    st.sidebar.metric("Citations failed", st.session_state.citations_failed)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("debug"):
                render_debug(msg["debug"])

    user_input = st.chat_input("Ask about the reviews...")
    if user_input:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = process_turn(user_input, history, use_memory, corpus)

        if result["citation_check"] is not None:
            st.session_state.citations_checked += 1
            if not result["citation_check"]["passed"]:
                st.session_state.citations_failed += 1
                log_citation_failure(user_input, result)

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result["citation_check"] is not None and not result["citation_check"]["passed"]:
                st.error(f"Citation check FAILED — invented review id(s) not in the retrieved "
                         f"context: {result['citation_check']['invented_ids']}")
            render_debug(result)

        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "debug": result})


if __name__ == "__main__":
    main()
