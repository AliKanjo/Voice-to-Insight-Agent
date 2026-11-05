"""
RAG (Retrieval-Augmented Generation) layer for Voice-to-Insight Agent.

Allows users to speak natural queries about their past recordings, meetings, and tasks.
Retrieves relevant notes via SQLite FTS5 + normalized matching, then synthesizes a grounded answer.
"""

import json
from typing import Any

import arabic_norm
import db
import understand

RAG_SYSTEM_PROMPT = """You are an intelligent voice assistant answering questions based STRICTLY on the user's recorded notes, tasks, and summaries provided in the context below.

Rules:
1. Answer concisely and conversationally (1-3 sentences) suitable for being spoken out loud.
2. Respond in the SAME language the user asked in (Arabic in -> Arabic out, English in -> English out, etc.).
3. If the answer cannot be found in the provided context, state politely that there are no recorded notes answering that question.
4. Do NOT make up facts not present in the context.
"""


def answer_query(question: str, client: Any = None, model: str = None) -> dict:
    """
    Perform retrieval over past voice notes and generate a grounded answer.
    """
    clean_q = arabic_norm.clean_transcript(question)
    matching_notes = db.search_notes(clean_q, limit=6)

    # If few or no direct matches, also include recent tasks and notes as context
    if len(matching_notes) < 3:
        recent = db.get_history(limit=5)
        seen_ids = {n["id"] for n in matching_notes}
        for r in recent:
            if r["id"] not in seen_ids:
                matching_notes.append(r)
                seen_ids.add(r["id"])

    # Format context for the LLM
    context_lines = []
    citations = []
    for note in matching_notes:
        note_id = note["id"]
        created = note.get("created_at", "")[:16].replace("T", " ")
        intent = note.get("intent", "note")
        transcript = note.get("transcript", "")
        raw_data = note.get("data") or {}

        line = f"[Note #{note_id} | {created} | Type: {intent}]: {transcript}"
        if intent == "task" and raw_data:
            line += f" -> Task: {raw_data.get('title')} (Due: {raw_data.get('due_date')}, Priority: {raw_data.get('priority')})"
        elif intent == "summary" and raw_data:
            line += f" -> Summary: {raw_data.get('summary_text')}"

        context_lines.append(line)
        citations.append({
            "id": note_id,
            "intent": intent,
            "transcript": transcript,
            "created_at": created,
        })

    context_str = "\n".join(context_lines) if context_lines else "(No notes recorded yet)"

    user_prompt = f"""User Question: {clean_q}

Recorded Context:
{context_str}

Provide a direct answer:"""

    try:
        if client is None or model is None:
            client, model = understand._resolve_client()

        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        answer_text = f"Could not generate answer: {exc}"

    return {
        "question": clean_q,
        "answer": answer_text,
        "citations": citations[:4],
        "context_count": len(citations),
    }

