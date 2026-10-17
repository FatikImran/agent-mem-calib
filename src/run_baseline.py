import json
from src.data.loader import load_tasks
from src.agent.graph import agent_app
from src.eval.parameter_f1 import compute_f1

def main():
    tasks = list(load_tasks())
    # Run on first 20 tasks for time purposes
    eval_tasks = tasks[:20]
    
    predictions = []
    golds = []
    
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
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent_app.invoke(initial_state)
            
        pred = result["final_tool_call"]
        predictions.append(pred or {})
        
        gold_call = task.raw.get("tool_call", {})
        gold_args = gold_call.get("arguments", {})
        if isinstance(gold_args, str):
            try:
                gold_args = json.loads(gold_args)
            except:
                pass
                
        golds.append({"arguments": gold_args})
        
        print("Done" if pred else "Failed to parse")

    print("\n--- Evaluation Results ---")
    precision, recall, f1 = compute_f1(predictions, golds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()