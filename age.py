import json
import os
import re
import subprocess
import tempfile
import time
import numpy as np
import openai
import faiss
from pymongo import MongoClient
import concurrent.futures
import importlib
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def profile_func(func):
    """Simple profiling decorator to record the execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[PROFILE] {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

# ─── Python Function Extraction ─────────────────────────────────────────────

def get_module_exports(module_name):
    """
    Get the exported functions/classes from a specified module.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"❌ Failed to import module {module_name}: {e}")
        return []
    exports = getattr(module, '__all__', None)
    if exports is None:
        exports = dir(module)
    records = []
    for name in exports:
        try:
            obj = getattr(module, name)
            doc = obj.__doc__
            if doc:
                doc = doc.strip().split('\n')[0]
            else:
                doc = "No description."
            records.append({
                "name": name,
                "package": module_name,
                "description": doc,
                "language": "Python"
            })
        except Exception:
            continue
    return records

def extract_ml_functions(ml_modules):
    """
    Iterate through multiple machine learning modules and extract all exported function/class information (Python).
    """
    all_records = []
    for module in ml_modules:
        try:
            records = get_module_exports(module)
            print(f"✅ Extracted {len(records)} records from module {module}")
            all_records.extend(records)
        except Exception as e:
            print(f"❌ Error extracting from module {module}: {e}")
    return all_records

# ─── R Function Extraction Helper Functions ─────────────────────────────────────────────

def remove_backspaces(s):
    """
    Simulate backspace behavior: remove the previous character when encountering a backspace character '\x08'.
    """
    result = []
    for char in s:
        if char == '\x08':
            if result:
                result.pop()
        else:
            result.append(char)
    return "".join(result)

def extract_description(help_text):
    """
    Extract the 'Description' section from help text:
    Clean the backspace characters, use regex to extract text between "Description:" and "Usage:",
    and compress extra whitespace into single spaces.
    """
    clean_text = remove_backspaces(help_text)
    pattern = r"Description:\s*(.*?)\s*Usage:"
    match = re.search(pattern, clean_text, re.DOTALL)
    if match:
        description_text = match.group(1)
        return " ".join(description_text.split())
    else:
        return ""

def extract_r_functions_from_packages(packages):
    """
    Iterate through predefined R packages and extract the name and description information of all functions.
    """
    r_records = []
    for pkg in packages:
        try:
            importr(pkg)
        except Exception as e:
            print(f"Error loading R package {pkg}: {e}")
            continue

        package_env = "package:" + pkg
        func_names = robjects.r('ls("%s")' % package_env)
        for func in func_names:
            func = str(func)
            try:
                # Determine whether the object is a function
                is_fun = robjects.r('is.function(get("%s", envir=as.environment("package:%s")))' % (func, pkg))[0]
            except Exception as e:
                is_fun = False
            if not is_fun:
                continue

            try:
                help_cmd = 'capture.output(tools:::Rd2txt(utils:::.getHelpFile(help("%s", package="%s"))))' % (func, pkg)
                help_lines = robjects.r(help_cmd)
                help_text = "\n".join([str(line) for line in help_lines])
                description = extract_description(help_text)
            except Exception as e:
                description = ""
            r_records.append({
                "name": func,
                "package": pkg,
                "description": description,
                "language": "R"
            })
        print(f"✅ Extracted {len(r_records)} records from R package {pkg}")
    return r_records

# ─── Database and FAISS Operations ─────────────────────────────────────────────

class Age:
    def __init__(self,
                 mongodb_uri="mongodb://localhost:27017/",
                 db_name="ai_agent_db",
                 openai_api_key=None):
        """
        Initialize MongoDB, FAISS vector index, and OpenAI API Key (if provided),
        and update the database by extracting functions from predefined Python and R modules.
        """
        if openai_api_key is not None:
            openai.api_key = openai_api_key

        # Initialize MongoDB
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db["packages"]

        # Drop all indexes (except the default _id index)
        self.collection.drop_indexes()
        # Create the necessary unique index
        self.collection.create_index("name", unique=True)
        print("✅ Connected to MongoDB database")

        # Clear the database and update data (including functions from Python and R)
        self.clear_database()
        self.update_database()

        # Initialize FAISS index and ID mapping (dimension 1536)
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_index = faiss.IndexIDMap(self.index)
        self.id_to_mongo_id = {}
        self.embedding_cache = {}

        # Load function records from MongoDB and build the FAISS index
        self.load_functions()

        print("✅ Age class initialized successfully")

    def clear_database(self):
        """Clear all function records from MongoDB (use with caution)."""
        self.collection.delete_many({})
        print("⚠️ All function data has been cleared from MongoDB")

    def update_database(self):
        """
        Extract function records from predefined Python modules and R packages,
        and insert or update the records into MongoDB.
        """
        # Extract Python functions
        python_modules = [
            "sklearn.ensemble",
            "sklearn.linear_model",
            "sklearn.tree",
            "sklearn.cluster",
            "sklearn.decomposition",
            "sklearn.preprocessing",
            "sklearn.model_selection",
            "sklearn.svm",
            "sklearn.neural_network",
            "sklearn.feature_extraction.text"
        ]
        py_functions = extract_ml_functions(python_modules)
        for func in py_functions:
            self.collection.update_one({"name": func["name"], "package": func["package"]},
                                         {"$set": func}, upsert=True)
        # Extract R functions
        r_packages = ["e1071", "caret", "MASS", "stats", "randomForest", "cluster"]
        r_functions = extract_r_functions_from_packages(r_packages)
        for func in r_functions:
            self.collection.update_one({"name": func["name"], "package": func["package"]},
                                         {"$set": func}, upsert=True)
        total = self.collection.count_documents({})
        print(f"✅ Database updated, total records: {total}")

    def get_embedding(self, text):
        """Call the OpenAI API to get the text embedding vector and use cache to reduce duplicate calls."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.embedding_cache[text] = embedding
        return embedding

    @profile_func
    def load_functions(self):
        """
        Load all function records from MongoDB, generate embedding vectors for each record,
        use cache to avoid duplicate computation, add them to the FAISS index via IndexIDMap,
        and record the mapping between custom IDs and MongoDB _id.
        """
        print("📌 Loading functions from MongoDB...")
        try:
            functions = list(
                self.collection.find({}, {"name": 1, "package": 1, "description": 1, "language": 1})
            )
            print(f"✅ Retrieved {len(functions)} functions from MongoDB.")
            if not functions:
                print("⚠️ No functions found in the database. Ensure data has been inserted.")
                return

            # Attempt to load cache (saved as a JSON file)
            cache_file = "embedding_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                self.embedding_cache = {k: np.array(v, dtype=np.float32) for k, v in cache_data.items()}
                print("✅ Loaded embedding cache from file.")
            else:
                self.embedding_cache = {}

            def get_embedding_for_func(func):
                key = f"{func['package']}::{func['name']}"
                if key in self.embedding_cache:
                    embedding = self.embedding_cache[key]
                else:
                    func_text = f"{func['name']}: {func['description']}"
                    embedding = self.get_embedding(func_text)
                    self.embedding_cache[key] = embedding
                return func, embedding

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(get_embedding_for_func, functions))

            for idx, (func, embedding) in enumerate(results):
                unique_id = np.int64(idx)
                self.id_index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([unique_id]))
                self.id_to_mongo_id[unique_id] = func["_id"]

            print(f"✅ Successfully loaded {len(functions)} functions into FAISS with ID mapping.")

            # Save the updated cache to a file (convert numpy arrays to lists)
            serializable_cache = {k: embedding.tolist() for k, embedding in self.embedding_cache.items()}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_cache, f, indent=4)
            print("✅ Embedding cache saved to file.")

        except Exception as e:
            print(f"❌ Error while retrieving functions from MongoDB: {e}")

    @profile_func
    def search(self, query, top_k=2):
        """
        Search for the vectors most similar to the query in the FAISS index,
        then map the returned custom IDs back to the MongoDB _id to get the full function records.
        """
        query_vector = np.array([self.get_embedding(query)], dtype=np.float32)
        distances, ids = self.id_index.search(query_vector, top_k)
        results = []
        for unique_id in ids[0]:
            if unique_id == -1:
                continue
            mongo_id = self.id_to_mongo_id.get(unique_id)
            if mongo_id:
                record = self.collection.find_one({"_id": mongo_id})
                if record:
                    results.append(record)
        return results if results else None

    def fill_parameters_with_openai(self, function_data):
        """
        Call the OpenAI API to generate a complete parameter list based on the function record,
        and return the filled parameter description text.
        """
        function_name = function_data["name"]
        package = function_data["package"]
        description = function_data["description"]
        prompt = f"""
    You are an AI assistant that helps to fill in missing arguments for a {function_name}.
    The function information is as follows:
    Please provide a detailed, complete parameter list (with explanations) that would be appropriate for calling this function in a machine learning context.
    Please provide an single line example with respect to call the selected function.
    Output only the parameter list and the example in plain text.
    """
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a parameter assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        filled_parameters = response.choices[0].message.content.strip()
        return filled_parameters

    @profile_func
    def generate_code_with_openai(self, function_data):
        """
        Call the OpenAI API to generate complete code in the corresponding language (Python or R) based on the function record.
        """
        function_name = function_data["name"]
        package = function_data["package"]
        language = function_data["language"]

        if language.lower() == "python":
            prompt = f"""
            You are an AI assistant that generates Python code for machine learning tasks.
            Please generate a complete Python script that:
            - Uses `{package}.{function_name}` for training
            - Loads the Iris dataset (`sklearn.datasets.load_iris`)
            - Splits data using `train_test_split`
            - Trains the model and makes predictions
            - Prints the first 5 predicted labels
            The script should be executable with all necessary imports.
            """
        elif language.lower() == "r":
            prompt = f"""
            You are an AI assistant that generates R code for statistical computing.
            Please generate a complete R script that:
            - Uses `{package}::{function_name}` for training
            - Loads the iris dataset
            - Splits data into train/test sets
            - Trains the model and makes predictions
            - Prints the first 5 predicted labels
            The script should be executable with all necessary library calls.
            """
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_output = response.choices[0].message.content
        match = re.search(r"```(python|r)\n(.*?)```", raw_output, re.DOTALL)
        if match:
            return match.group(2)
        else:
            return raw_output

    @profile_func
    def execute_python_code(self, code):
        """
        Execute the generated Python code using subprocess and return the result or error message.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            result = subprocess.run(["python", temp_file_path], capture_output=True, text=True)
            os.remove(temp_file_path)

            if result.returncode == 0:
                return f"✅ Code executed successfully!\nOutput:\n{result.stdout}"
            else:
                return f"❌ Failure!\nError message:\n{result.stderr}"
        except Exception as e:
            return f"❌ Error occurs when executing the code: {e}"

    @profile_func
    def execute_r_code(self, code):
        """
        Execute the generated R code using rpy2 and return the result or error message.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            robjects.r['source'](temp_file_path)
            os.remove(temp_file_path)
            return "✅ R code executed successfully!"
        except Exception as e:
            return f"❌ Error occurs when executing the R code via rpy2: {e}"

    @profile_func
    def run_query(self, top_k=5):
        """
        Enter an interactive query session:
          1. Continuously prompt the user for a query (or "exit" to quit).
          2. For each query, search for matching functions in the FAISS index.
          3. If multiple matches are found, prompt the user to choose one.
          4. Call OpenAI to generate corresponding code.
          5. Execute the generated code based on the language and output the result.
        """
        while True:
            query_text = input("\nEnter your query (or type 'exit' to quit): ")
            if query_text.strip().lower() == "exit":
                print("Exiting the query session.")
                break

            search_results = self.search(query_text, top_k=top_k)
            if search_results and isinstance(search_results, list):
                print("\n🔍 **Matched Functions:**")
                for i, func in enumerate(search_results):
                    print(f"{i + 1}. {func['name']} (Package: {func['package']}, Language: {func['language']})")

                if len(search_results) > 1:
                    choice = input("\n👉 Enter the number of the function you want to use: ")
                    try:
                        selected_func = search_results[int(choice) - 1]
                    except (IndexError, ValueError):
                        print("❌ Invalid selection. Skipping this query...")
                        continue
                else:
                    selected_func = search_results[0]

                filled_function = self.fill_parameters_with_openai(selected_func)
                print(f"\n✅ Filled function parameters:\n{filled_function}\n")

                answer = input("Do you want to generate instance code based on the above filled arguments? (Y/N): ")
                if answer.strip().lower() == "y":
                    code = self.generate_code_with_openai(selected_func)
                    print("\n📌 Generated Code:\n", code)

                    if selected_func["language"].lower() == "python":
                        print("\n🚀 Executing Python Code...\n")
                        result = self.execute_python_code(code)
                    elif selected_func["language"].lower() == "r":
                        print("\n🚀 Executing R Code...\n")
                        result = self.execute_r_code(code)
                    else:
                        print(f"❌ Unsupported language: {selected_func['language']}")
                        result = None

                    if result:
                        print("\n📊 Execution Output:\n", result)
                else:
                    print("Code generation aborted by the user.")
            else:
                print("❌ No matching function found.")


# if __name__ == "__main__":
#     # Ensure that the OPENAI_API_KEY environment variable is set before running.
#     api_key = os.getenv("OPENAI_API_KEY")
#     age = Age(openai_api_key=api_key)
#     age.run_query()
