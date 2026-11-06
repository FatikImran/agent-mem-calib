"""Parse tool calls from structured messages or the local model text format."""
import json
import re


def parse_tool_call(msg):
    tool_calls = msg.get("tool_calls")
    final_call = None
    
    if tool_calls:
        function = tool_calls[0].get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if function.get("name") and isinstance(arguments, dict):
            final_call = {"name": function["name"], "arguments": arguments}

    if final_call is None:
        content = msg.get("content") or ""
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
                if isinstance(parsed, dict) and parsed.get("name") and isinstance(parsed.get("arguments"), dict):
                    final_call = {"name": parsed["name"], "arguments": parsed["arguments"]}
            except json.JSONDecodeError:
                pass
    return final_call
