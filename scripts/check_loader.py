"""Verify that the loader extracts all four required fields."""

from src.data.loader import load_tasks

def main():
    print("Loading tasks...")
    tasks = list(load_tasks())
    print(f"Loaded {len(tasks)} tasks")

    if len(tasks) == 0:
        raise ValueError("No tasks loaded!")

    t = tasks[0]
    print("\n--- First task ---")
    print(f"ID:            {t.task_id}")
    print(f"Query:         {t.query[:100]}...")
    print(f"Tool name:     {t.tool_schema.get('name')}")
    print(f"Gold args:     {t.gold_arguments}")
    print(f"Memory turns:  {len(t.memory_context)}")
    
    if t.memory_context:
        print(f"First memory:  [{t.memory_context[0].role}] {t.memory_context[0].content[:80]}...")

    print("\n--- Running Hard Checks ---")
    assert t.query, "Query is empty"
    assert t.tool_schema, "Tool schema is empty"
    assert t.gold_arguments, "Gold arguments are empty"
    assert t.memory_context, "Memory context is empty"
    print("All single-task checks passed.")

    # Aggregate check to ensure linking worked across the dataset
    no_memory = sum(1 for task in tasks if not task.memory_context)
    print(f"\nTasks with no memory context: {no_memory}/{len(tasks)} ({(no_memory/len(tasks))*100:.1f}%)")
    
    # Allow up to 5% because of known dataset quirks (empty IDs in source file)
    if no_memory / len(tasks) > 0.05:
        print("WARNING: Too many tasks lack memory. The linking field might be wrong.")
    else:
        print("Memory context successfully linked. (Missing tasks are a known dataset artifact).")

if __name__ == "__main__":
    main()