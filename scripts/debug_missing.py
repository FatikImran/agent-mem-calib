from src.data.loader import load_tasks

tasks = list(load_tasks())
missing = [t for t in tasks if not t.memory_context]

print(f"Total missing: {len(missing)}\n")
for t in missing[:5]:
    print(f"Task: {t.task_id}")
    print(f"  Expected IDs: {t.raw.get('source_conversation_ids')}")
    print(f"  Query: {t.query[:60]}...\n")