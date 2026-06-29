"""Pre-download the Qwen3-Embedding-0.6B text encoder (transformers caches it; optional)."""
from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
print("Qwen3-Embedding-0.6B cached.")
