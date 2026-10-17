from huggingface_hub import hf_hub_download
import os

os.makedirs("./models", exist_ok=True)

print("Downloading Qwen_Qwen3-4B-Q4_K_M.gguf...")
path = hf_hub_download(
    repo_id="bartowski/Qwen_Qwen3-4B-GGUF",
    filename="Qwen_Qwen3-4B-Q4_K_M.gguf",  # <-- Correct filename with Qwen_ prefix
    local_dir="./models"
)
print(f"Download complete! File saved to: {path}")