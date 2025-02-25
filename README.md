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

## 3. Example Queries and Expected Outputs
Query: Train a decision tree model
System Response:
- Matches a relevant function (e.g., DecisionTreeClassifier from sklearn.tree).
- Generates Python code snippet for reading a dataset, training the model, and printing predictions.
- Asks if you want to execute the code. On execution, displays the console output or an error.

## 4.Expanding the Package Database
1. **Add More Entries:
Edit or create new JSON files (e.g., py_functions.json, r_functions.json) with additional function records. Each record typically has:
json
{
  "name": "FunctionName",
  "language": "python|r",
  "package": "sklearn.xxx / stats / etc.",
  "description": "A short description of what the function does."
}
2. **Auto Insert:
On startup, the system will read those JSON files and insert them into MongoDB, then update the FAISS index.
3. **Scale:
You can maintain hundreds of function records spanning multiple packages, simply by appending them to the JSON files (or inserting them via the agent’s code).