from pathlib import Path
import contextlib
import io
import json
from src.data.loader import load_tasks
from src.agent.graph import agent_app
from src.eval.pilot_outcomes import summarize

def main():
    tasks = list(load_tasks())
    # Run on first 20 tasks for time purposes
    eval_tasks = tasks[:20]
    
    records = []
    
    for i, task in enumerate(eval_tasks):
        print(f"[{i+1}/{len(eval_tasks)}] Running {task.task_id}...", end=" ", flush=True)
        
        initial_state = {
            "query": task.query,
            "tool_schema": task.tool_schema,
            "memory_context": [m.model_dump() for m in task.memory_context],
            "messages": [],
            "final_tool_call": None
        }
        
        # We will suppress the printing inside the graph for the batch run
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent_app.invoke(initial_state)
            
        pred = result["final_tool_call"]
        
        gold_args = task.gold_arguments
        parsed = isinstance(pred, dict) and isinstance(pred.get("arguments"), dict)
        records.append({
            "task_id": task.task_id,
            "parsed": parsed,
            "prediction": pred,
            "gold_arguments": gold_args,
        })

        print("Done" if parsed else "Failed to parse")

    output = Path("results/phase1-pilot.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summarize(records), indent=2))

if __name__ == "__main__":
    main()