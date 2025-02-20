import os
import openai
import faiss
import numpy as np
from pymongo import MongoClient
import subprocess
import tempfile
import re

# just for test, in reality you should hide the api key(like in local)
openai.api_key = "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"


class SemanticSearch:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="ai_agent_db"):
        """
        初始化 MongoDB 和 FAISS 索引
        """
        # 连接 MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["ml_functions"]

        # 创建 FAISS 索引
        self.index = faiss.IndexFlatL2(1536)  # OpenAI Embeddings 维度是 1536
        self.function_map = {}

        # 预加载数据库中的所有函数
        self.load_functions()

    def get_embedding(self, text):
        """
        获取文本的 OpenAI 向量嵌入
        """
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def load_functions(self):
        """
        Loads function descriptions from MongoDB and stores them in FAISS.
        """
        print("📌 Loading functions from local DB...")

        try:
            functions = list(self.collection.find({}, {"_id": 0, "name": 1, "package": 1, "description": 1}))
            print(f"✅ Retrieved {len(functions)} functions from MongoDB.")

            if len(functions) == 0:
                print("⚠️ No functions found in the database. Ensure the data has been inserted.")

            for idx, func in enumerate(functions):
                func_text = func["description"]
                embedding = self.get_embedding(func_text)

                # Store in FAISS
                self.index.add(np.array([embedding], dtype=np.float32))
                self.function_map[idx] = func

            print(f"✅ Successfully loaded {len(functions)} functions into FAISS.")

        except Exception as e:
            print(f"❌ Error while retrieving functions from MongoDB: {e}")

    def search(self, query, top_k=1):
        """
        在 FAISS 中查找最相似的函数
        """
        query_vector = np.array([self.get_embedding(query)], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:  # 获取 top_k 结果
            if idx in self.function_map:
                results.append(self.function_map[idx])

        return results[0] if results else None


class CodeGenerator:
    def __init__(self):
        """
        Generate the code
        """
        if not openai.api_key:
            raise ValueError("❌ OpenAI API Key is required. Please, check your settings")

    import openai

    def generate_code_with_openai(self,function_data):
        """
        Uses OpenAI's updated API to generate Python code.
        """
        client = openai.OpenAI(api_key="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")  # ✅ Correct way to initialize OpenAI client

        function_name = function_data["name"]
        package = function_data["package"]

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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python machine learning assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        raw_output=response.choices[0].message.content

        # ✅ Extract only the Python code using regex
        match = re.search(r"```python\n(.*?)```", raw_output, re.DOTALL)
        if match:
            return match.group(1)  # Extract only the valid Python code
        else:
            return raw_output  # Return original if no match (fallback)

    def execute_python_code(self, code):
        """
        use subprocess to execute python code
        """
        try:
            # create a temp py file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # use subprocess to run the py code
            result = subprocess.run(["python", temp_file_path], capture_output=True, text=True)

            # delete the temp file
            os.remove(temp_file_path)

            # return the result
            if result.returncode == 0:
                return f"✅ Code execute successfully！\nOutput:\n{result.stdout}"
            else:
                return f"❌ Failure！\nError message:\n{result.stderr}"

        except Exception as e:
            return f"❌ Error occurs when executing the code: {e}"




