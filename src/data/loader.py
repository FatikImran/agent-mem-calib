"""Loader for Mem2ActBench tasks."""

import json
from pathlib import Path
from typing import Iterator
from pydantic import BaseModel

# Adjust this path if your folder structure is different
DATA_DIR = Path("data/Mem2ActBench/Mem2ACTBench_small")


class MemoryTurn(BaseModel):
    role: str
    content: str
    source_id: str


class Task(BaseModel):
    task_id: str
    query: str
    tool_schema: dict
    gold_arguments: dict
    memory_context: list[MemoryTurn]
    raw: dict


def load_conversations() -> dict[str, list[MemoryTurn]]:
    """Load all conversation turns indexed by their source_id."""
    conv_path = DATA_DIR / "Mem2ACT_conversation.jsonl"
    source_to_turns: dict[str, list[MemoryTurn]] = {}

    with conv_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for turn in record.get("turns", []):
                source_id = turn.get("source_id")
                if source_id:
                    if source_id not in source_to_turns:
                        source_to_turns[source_id] = []
                    source_to_turns[source_id].append(
                        MemoryTurn(
                            role=turn.get("role", "unknown"),
                            content=turn.get("content", ""),
                            source_id=source_id
                        )
                    )
    return source_to_turns


def load_tasks() -> Iterator[Task]:
    """Yield Task objects from the QA dataset, attaching memory context."""
    qa_path = DATA_DIR / "qa_dataset.jsonl"
    source_to_turns = load_conversations()

    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            
            # 1. Extract memory context based on the source IDs
            source_ids = record.get("source_conversation_ids", [])
            memory_context = []
            for sid in source_ids:
                if sid in source_to_turns:
                    memory_context.extend(source_to_turns[sid])
            
            # Sort memory turns to maintain chronological order (optional but good for context)
            # We don't have timestamps here, so we just append them in the order they appear.

            # 2. Extract gold arguments
            tool_call = record.get("tool_call", {})
            gold_args = tool_call.get("arguments", {})
            if isinstance(gold_args, str):
                try:
                    gold_args = json.loads(gold_args)
                except json.JSONDecodeError:
                    pass

            yield Task(
                task_id=record.get("qa_id", ""),
                query=record.get("query", ""),
                tool_schema=record.get("target_tool_schema", {}),
                gold_arguments=gold_args,
                memory_context=memory_context,
                raw=record
            )