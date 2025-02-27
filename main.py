from age import Age
import os

api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
age = Age(openai_api_key=api_key)
age.run_query()
