from age import Age
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY1")
if not api_key:
    raise Exception("API key not found in environment variables")
# # Ensure that the OPENAI_API_KEY environment variable is set before running.
age = Age(openai_api_key=api_key)
# # If you want to use test cases to test the accuracy, use this line.
age.benchmark_query_accuracy("benchmark_tests.json", 10)

# # If you want to use the function of this python class, use this line.
age.run_query()