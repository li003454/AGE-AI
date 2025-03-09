import json
import time
import os
import numpy as np
import openai
import faiss
from annotated_types import test_cases
from pymongo import MongoClient
import concurrent.futures
from dotenv import load_dotenv


from functions import (
    extract_ml_functions,
    extract_r_functions_from_packages
)

from utils import performance_metrics, profile_func, print_performance_metrics
from openai_utils import fill_parameters, generate_code
from code_executor import execute_python_code, execute_r_code





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
        self.mongodb_uri = mongodb_uri
        self.openai_api_key = openai_api_key
        self.db_name = db_name


        openai.api_key = self.openai_api_key
        if openai_api_key is not None:
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
            self.summarize_database()

    def summarize_database(self):
        """
        Summarize the database by printing the total number of function records,
        as well as a breakdown by language.
        """
        total = self.collection.count_documents({})
        python_count = self.collection.count_documents({"language": "Python"})
        r_count = self.collection.count_documents({"language": "R"})


        print("\n📊 Database Summary:")
        print(f"   Total function records: {total}")
        print(f"   Python functions: {python_count}")
        print(f"   R functions: {r_count}")

    @profile_func
    def clear_database(self):
        """Clear all function records from MongoDB (use with caution)."""
        self.collection.delete_many({})
        print("⚠️ All function data has been cleared from MongoDB")

    @profile_func
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
            "lightgbm",  # LightGBM（轻量级梯度提升）
            "catboost",  # CatBoost 梯度提升库
            # "sklearn.pipeline",  # Added for pipeline utilities
            # "sklearn.metrics",  # Added for performance metrics
            # "sklearn.naive_bayes",  # Added for Naive Bayes models
            # "sklearn.manifold",  # Added for manifold learning
            # "sklearn.gaussian_process"  # Added for Gaussian process models
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


    @profile_func
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
    def search(self, query, top_k=5):
        # 生成查询嵌入
        query_embedding = self.get_embedding(query)
        # 对嵌入向量归一化（余弦相似度）
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query_vector = np.array([query_embedding], dtype=np.float32)

        # 使用内积作为度量
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



    @profile_func
    def run_query(self, top_k=5):
        """
        Enter an interactive query session:
          1. Continuously prompt the user for a query (or "exit" to quit).
          2. For each query, search for matching functions in the FAISS index.
          3. If multiple matches are found, prompt the user to choose one.
          4. Call OpenAI to generate corresponding code.
          5. Execute the generated code based on the language and output the result.
          6. If the user inputs "performance", print the performance metrics.
        """
        while True:
            query_text = input("\nEnter your query (or type 'exit' to quit, 'performance' to show metrics): ")
            if query_text.strip().lower() == "exit":
                print("Exiting the query session.")
                break
            elif query_text.strip().lower() == "performance":
                print_performance_metrics()
                continue

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

                filled_function = fill_parameters(self.openai_api_key,selected_func)
                print(f"\n✅ Filled function parameters:\n{filled_function}\n")

                answer = input("Do you want to generate instance code based on the above filled arguments? (Y/N): ")
                if answer.strip().lower() == "y":
                    code = generate_code(self.openai_api_key,selected_func)
                    print("\n📌 Generated Code:\n", code)

                    if selected_func["language"].lower() == "python":
                        print("\n🚀 Executing Python Code...\n")
                        result = execute_python_code(code)
                    elif selected_func["language"].lower() == "r":
                        print("\n🚀 Executing R Code...\n")
                        result = execute_r_code(code)
                    else:
                        print(f"❌ Unsupported language: {selected_func['language']}")
                        result = None

                    if result:
                        print("\n📊 Execution Output:\n", result)
                else:
                    print("Code generation aborted by the user.")
            else:
                print("❌ No matching function found.")

# ─── Benchmark to test the code ─────────────────────────────────────────────

    def benchmark_query_accuracy(self, test_cases, top_k=5):
        """
        Evaluate the query accuracy over a set of benchmark test cases.

        Each test case is expected to be a dictionary containing a "query" string and an "expected_function" string.
        The test is considered passed if any function name in the search results contains the expected function name (case-insensitive).

        Parameters:
          - test_cases: A file path (string) to a JSON file containing the test cases.
          - top_k: The number of search results to return for each query (default is 5).

        This method prints the result for each test case and the overall accuracy.
        """
        benchmark_file = test_cases
        with open(benchmark_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        passed = 0
        total = len(test_cases)

        for case in test_cases:
            query = case["query"]
            expected_func = case["expected_function"].lower()
            results = self.search(query, top_k=top_k)
            # The test passes if any of the returned function names (case-insensitive) contain the expected function keyword.
            found = any(expected_func in func["name"].lower() for func in results) if results else False

            status = "Pass" if found else "Fail"
            if found:
                passed += 1
            print(f"Query: '{query}' | Expected Function: '{case['expected_function']}' | Status: {status}")

        accuracy = (passed / total) * 100 if total else 0
        print(f"\nTotal Test Cases: {total}, Passed: {passed}, Accuracy: {accuracy:.2f}%")
