from age import Age
import os

api_key = os.getenv("OPENAI_API_KEY")
age = Age(openai_api_key=api_key)
query = input("Enter your query: ")
age.run_query(query)