from llama_cpp import Llama
import json

# Load the model
llm = Llama(
    model_path="./models/Qwen_Qwen3-4B-Q4_K_M.gguf",  # <-- Updated filename
    n_ctx=2048,          # Keep context small for 4GB VRAM
    n_gpu_layers=-1,     # Offload all layers to GPU
    verbose=False,
)

# Test a tool call
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is the weather in New York for the next 7 days?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "GetWeatherForecast",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of days"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto",
)

print(json.dumps(response["choices"][0]["message"], indent=2))