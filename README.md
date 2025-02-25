# AGE-AI
AGE (Auto Generating and Executing) AI Agent for Python/R


## 1. Project Overview
The **AGE AI Agent** is designed to simplify machine learning and data analysis tasks by automatically:
1. **Retrieving** relevant Python or R functions from a database (MongoDB + FAISS).
2. **Generating code** tailored to the user’s request (via OpenAI GPT).
3. **Executing** that code (Python or R) and returning results or errors.

This approach reduces repetitive coding tasks, allowing data scientists and developers to prototype faster.

## 2. Installation and Dependency Setup
1. **Python 3.8+ (recommended) if not already available.
2. **Install Dependencies: MongoDB: Install and start MongoDB on localhost:27017.
3. **Python Packages: openai, pymongo, faiss, rpy2
4. **R execution(optional) Install R (4.x or higher).
Ensure environment variable R_HOME is set (especially on Windows).

