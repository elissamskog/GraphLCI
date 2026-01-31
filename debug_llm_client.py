
import os
import logging
from llm_client import get_client, _load_config

logging.basicConfig(level=logging.INFO)

print("--- DEBUGGING LLM CLIENT ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Config Load Check: {_load_config()}")

try:
    client = get_client()
    print(f"Client type: {type(client)}")
except Exception as e:
    print(f"Error getting client: {e}")

try:
    print("Testing Generation...")
    response = client.generate("Hello, are you working?", num_ctx=100)
    print(f"Response: {response}")
except Exception as e:
    print(f"Error generating: {e}")
