"""Streamlit app — Step 2: extract validated tickets from cleaned reviews.

Reads data/clean_reviews.json, calls the configured LLM provider (PROVIDER
in .env) using that provider's native structured-output feature, then
re-validates every response in pure Python before accepting it.

Run with: streamlit run "2_extract_tickets.py"
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CLEAN_PATH = Path("data/clean_reviews.json")
TICKETS_PATH = Path("data/tickets.json")

CATEGORIES = ["shipping", "product_quality", "customer_service", "billing", "other"]
SEVERITIES = ["low", "medium", "high"]
TICKET_KEYS = {"id", "category", "severity", "refund_requested", "summary"}

# Shared shape across all three providers: OpenAI's
# response_format.json_schema.schema and Anthropic's tool input_schema both
# accept this JSON-Schema dict as-is (OpenAI strict mode requires
# additionalProperties: False). Gemini's response_schema uses the same
# properties/required but rejects the additionalProperties key outright —
# see GEMINI_TICKET_SCHEMA below.
TICKET_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "The review id, copied through unchanged."},
        "category": {"type": "string", "enum": CATEGORIES},
        "severity": {"type": "string", "enum": SEVERITIES},
        "refund_requested": {"type": "boolean"},
        "summary": {"type": "string", "description": "One sentence summarizing the review."},
    },
    "required": ["id", "category", "severity", "refund_requested", "summary"],
    "additionalProperties": False,
}

# Gemini's REST API rejects "additionalProperties" in response_schema with
# "Unknown name additional_properties: Cannot find field" — confirmed live.
# Same schema minus that one key.
GEMINI_TICKET_SCHEMA = {k: v for k, v in TICKET_JSON_SCHEMA.items() if k != "additionalProperties"}

GEMINI_MODEL = "gemini-3.6-flash"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

def _build_prompt(review, correction_context=None):
    prompt = (
        "Extract a support ticket from this customer review.\n"
        f'Review id: "{review["id"]}"\n'
        f'Review text: "{review["text"]}"\n\n'
        f'The "id" field of your answer must be exactly "{review["id"]}".\n'
        f"category must be one of: {', '.join(CATEGORIES)}.\n"
        f"severity must be one of: {', '.join(SEVERITIES)}.\n"
        "summary must be a single sentence."
    )
    if correction_context:
        errors_text = "\n".join(f"- {e}" for e in correction_context["errors"])
        prompt += (
            "\n\nYour previous answer was invalid:\n"
            f"{json.dumps(correction_context['previous_json'])}\n\n"
            f"Validation errors:\n{errors_text}\n\n"
            "Return a corrected JSON object that fixes every error above."
        )
    return prompt


# --------------------------------------------------------------------------
# Provider branches — each SDK is imported lazily, inside its own function,
# so a missing package never breaks startup for the other two branches.
# --------------------------------------------------------------------------

def _extract_ticket_gemini(review, correction_context, api_key):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(review, correction_context),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_TICKET_SCHEMA,
        ),
    )
    return json.loads(response.text)


def _extract_ticket_openai(review, correction_context, api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": _build_prompt(review, correction_context)}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ticket", "schema": TICKET_JSON_SCHEMA, "strict": True},
        },
    )
    return json.loads(response.choices[0].message.content)


def _extract_ticket_anthropic(review, correction_context, api_key):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        tools=[{"name": "emit_ticket", "input_schema": TICKET_JSON_SCHEMA}],
        tool_choice={"type": "tool", "name": "emit_ticket"},
        messages=[{"role": "user", "content": _build_prompt(review, correction_context)}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Anthropic response contained no tool_use block")


def extract_ticket(review, correction_context=None):
    """Dispatch to the provider named by PROVIDER. Returns the raw parsed
    ticket dict — NOT yet validated."""
    provider = os.getenv("PROVIDER", "gemini").lower()
    if provider == "gemini":
        return _extract_ticket_gemini(review, correction_context, os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return _extract_ticket_openai(review, correction_context, os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        return _extract_ticket_anthropic(review, correction_context, os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown PROVIDER: {provider!r} (expected openai, anthropic, or gemini)")


# --------------------------------------------------------------------------
# Validation — pure Python, no SDK dependency, never trusts provider schema
# enforcement.
# --------------------------------------------------------------------------

def validate_ticket(obj, expected_id):
    """Return a list of human-readable errors; empty list means valid."""
    if not isinstance(obj, dict):
        return [f"ticket is not a JSON object (got {type(obj).__name__})"]

    errors = []
    missing = TICKET_KEYS - obj.keys()
    extra = obj.keys() - TICKET_KEYS
    for key in sorted(missing):
        errors.append(f"missing required key '{key}'")
    for key in sorted(extra):
        errors.append(f"unexpected extra key '{key}'")

    if "id" in obj:
        if not isinstance(obj["id"], str):
            errors.append(f"'id' must be a string, got {type(obj['id']).__name__}")
        elif obj["id"] != expected_id:
            errors.append(f"'id' is '{obj['id']}' but expected '{expected_id}'")

    if "category" in obj:
        if not isinstance(obj["category"], str):
            errors.append(f"'category' must be a string, got {type(obj['category']).__name__}")
        elif obj["category"] not in CATEGORIES:
            errors.append(f"'category' is '{obj['category']}', must be one of {CATEGORIES}")

    if "severity" in obj:
        if not isinstance(obj["severity"], str):
            errors.append(f"'severity' must be a string, got {type(obj['severity']).__name__}")
        elif obj["severity"] not in SEVERITIES:
            errors.append(f"'severity' is '{obj['severity']}', must be one of {SEVERITIES}")

    if "refund_requested" in obj and not isinstance(obj["refund_requested"], bool):
        errors.append(
            f"'refund_requested' must be a boolean, got {type(obj['refund_requested']).__name__}"
        )

    if "summary" in obj and not isinstance(obj["summary"], str):
        errors.append(f"'summary' must be a string, got {type(obj['summary']).__name__}")

    return errors


def process_review(review):
    """Call the model, validate, retry once on failure. Never loops.

    Returns (status, ticket_or_None, errors) with status one of
    PASSED / RETRIED / FAILED.
    """
    try:
        raw = extract_ticket(review)
    except Exception as e:
        return "FAILED", None, [f"API call failed: {e}"]

    errors = validate_ticket(raw, review["id"])
    if not errors:
        return "PASSED", raw, []

    try:
        corrected = extract_ticket(
            review, correction_context={"previous_json": raw, "errors": errors}
        )
    except Exception as e:
        return "FAILED", None, errors + [f"retry API call failed: {e}"]

    retry_errors = validate_ticket(corrected, review["id"])
    if not retry_errors:
        return "RETRIED", corrected, errors
    return "FAILED", None, retry_errors


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data
def load_clean_reviews():
    with open(CLEAN_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_existing_tickets():
    """Plain file read (no LLM call) — safe to call at module/session start."""
    if not TICKETS_PATH.exists():
        return [], []
    with open(TICKETS_PATH, encoding="utf-8") as f:
        tickets = json.load(f)
    # Reload can't distinguish a past RETRIED from PASSED since only
    # successful tickets are persisted to disk; both show as PASSED here.
    log = [{"id": t["id"], "status": "PASSED", "errors": ""} for t in tickets]
    return tickets, log


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Extract Tickets", layout="wide")
    st.title("Step 2 — Extract Tickets")

    if not CLEAN_PATH.exists():
        st.error(f"{CLEAN_PATH} not found — run Step 1 (1_clean_reviews.py) and click Save first.")
        st.stop()

    reviews = load_clean_reviews()
    provider = os.getenv("PROVIDER", "gemini")
    st.caption(f"Provider: **{provider}**  ·  {len(reviews)} cleaned reviews")

    if "tickets" not in st.session_state:
        st.session_state.tickets, st.session_state.log = load_existing_tickets()

    if st.button("Run extraction"):
        tickets, log = [], []
        progress = st.progress(0.0)
        for i, review in enumerate(reviews):
            status, ticket, errors = process_review(review)
            log.append({"id": review["id"], "status": status, "errors": "; ".join(errors)})
            if ticket is not None:
                tickets.append(ticket)
            progress.progress((i + 1) / len(reviews))

        tickets.sort(key=lambda t: t["id"])
        st.session_state.tickets = tickets
        st.session_state.log = log
        with open(TICKETS_PATH, "w", encoding="utf-8") as f:
            json.dump(tickets, f, indent=2, ensure_ascii=False)
        st.success(f"Extraction complete: {len(tickets)}/{len(reviews)} tickets saved to {TICKETS_PATH}")

    tickets = st.session_state.get("tickets", [])
    log = st.session_state.get("log", [])

    st.subheader("Successful tickets")
    if tickets:
        tickets_df = pd.DataFrame(tickets)
        st.dataframe(tickets_df, use_container_width=True)
        st.download_button(
            "Download tickets.csv",
            tickets_df.to_csv(index=False),
            file_name="tickets.csv",
            mime="text/csv",
        )
    else:
        st.write("No tickets yet — press Run extraction.")

    st.subheader("Validation log")
    if log:
        log_df = pd.DataFrame(log)
        counts = log_df["status"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("PASSED", int(counts.get("PASSED", 0)))
        c2.metric("RETRIED", int(counts.get("RETRIED", 0)))
        c3.metric("FAILED", int(counts.get("FAILED", 0)))
        st.dataframe(log_df, use_container_width=True)
    else:
        st.write("No log yet — press Run extraction.")

    st.subheader("Validator proof (no API calls)")
    st.caption("Runs validate_ticket against hand-written fixtures to prove it actually rejects bad input.")
    if st.button("Test validator"):
        fixtures = [
            ("valid ticket", {
                "id": "r001", "category": "shipping", "severity": "medium",
                "refund_requested": False, "summary": "Package never arrived.",
            }, "r001"),
            ("invalid category ('delivery')", {
                "id": "r002", "category": "delivery", "severity": "low",
                "refund_requested": True, "summary": "Wrong item shipped.",
            }, "r002"),
            ("invalid severity ('critical')", {
                "id": "r003", "category": "shipping", "severity": "critical",
                "refund_requested": False, "summary": "Box was crushed.",
            }, "r003"),
        ]
        for label, fixture, expected_id in fixtures:
            st.markdown(f"**{label}**")
            st.code(json.dumps(fixture, indent=2), language="json")
            errors = validate_ticket(fixture, expected_id)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.success("No errors — valid ticket.")


if __name__ == "__main__":
    main()
