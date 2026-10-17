import json
import re
from typing import TypedDict, Annotated, Sequence
import operator

from llama_cpp import Llama
from langgraph.graph import StateGraph, END

# 1. Define the State
class AgentState(TypedDict):
    query: str
    tool_schema: dict
    memory_context: list[dict]
    messages: Annotated[Sequence[dict], operator.add]
    final_tool_call: dict | None

# 2. Load the Model globally (so it doesn't reload on every graph step)
print("Loading model...")
llm = Llama(
    model_path="./models/Qwen_Qwen3-4B-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False,
)

# 3. Define the Nodes
def retrieve_memory(state: AgentState):
    """Simple baseline: just format the memory context into a string."""
    memory_str = ""
    for turn in state["memory_context"]:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        memory_str += f"[{role}]: {content}\n"
    
    sys_msg = {
        "role": "system", 
        "content": f"You are a helpful assistant. Here is the user's background context:\n{memory_str}\n\nUse this context to infer missing parameters if needed."
    }
    return {"messages": [sys_msg]}

def call_tool(state: AgentState):
    """Call the LLM and parse the tool call."""
    tool_def = [{
        "type": "function",
        "function": {
            "name": state["tool_schema"].get("name", "unknown"),
            "description": state["tool_schema"].get("description", ""),
            "parameters": state["tool_schema"].get("parameters", {})
        }
    }]
    
    messages = list(state["messages"])
    messages.append({"role": "user", "content": state["query"]})
    
    response = llm.create_chat_completion(
        messages=messages,
        tools=tool_def,
        tool_choice="auto",
        max_tokens=512,
        temperature=0.1
    )
    
    msg = response["choices"][0]["message"]
    
    # Print exactly what the model generated so we can debug
    print("\n--- RAW MODEL OUTPUT ---")
    print(msg)
    print("------------------------\n")
    
    tool_calls = msg.get("tool_calls")
    final_call = None
    
    if tool_calls:
        final_call = {
            "name": tool_calls[0]["function"]["name"],
            "arguments": json.loads(tool_calls[0]["function"]["arguments"])
        }
    else:
        content = msg.get("content", "")
        # Look for the <tool_call> tags specifically
        tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
        
        json_str = ""
        if tool_call_match:
            json_str = tool_call_match.group(1)
        else:
            # Fallback: just try to find any JSON object that has "name" and "arguments"
            json_match = re.search(r'(\{.*"name".*"arguments".*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

        if json_str:
            try:
                parsed = json.loads(json_str)
                if "name" in parsed and "arguments" in parsed:
                    final_call = {"name": parsed["name"], "arguments": parsed["arguments"]}
            except json.JSONDecodeError:
                pass
                
    return {"final_tool_call": final_call, "messages": [msg]}

# 4. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_memory", retrieve_memory)
workflow.add_node("call_tool", call_tool)

workflow.set_entry_point("retrieve_memory")
workflow.add_edge("retrieve_memory", "call_tool")
workflow.add_edge("call_tool", END)

agent_app = workflow.compile()