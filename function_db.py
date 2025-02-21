import json
from pymongo import MongoClient


class FunctionDatabase:
    """
    A class to manage machine learning function storage in MongoDB.
    This includes inserting, retrieving, and deleting function records.
    """

    def __init__(self, uri="mongodb://localhost:27017/", db_name="ai_agent_db"):
        """
        Initializes the MongoDB connection and selects the database and collection.
        Creates an index on the 'name' field for faster queries.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["ml_functions"]
        self.collection.create_index("name")  # Create an index on function names
        print("✅ Connected to MongoDB database")

    def insert_function(self, function_data: dict):
        """
        Inserts a single function record into MongoDB.
        :param function_data: A dictionary containing function details
        """
        existing = self.collection.find_one({"name": function_data["name"], "language": function_data["language"]})
        if not existing:
            result = self.collection.insert_one(function_data)
            print(f"✅ Inserted function: {function_data['name']} ({function_data['language']}) (ID: {result.inserted_id})")
        else:
            print(f"⚠️ Function {function_data['name']} ({function_data['language']}) already exists, skipping.")

    def insert_functions_from_json(self, json_files):
        """
        Reads multiple JSON files and inserts function records into MongoDB.
        :param json_files: List of JSON file paths
        """
        all_data = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):  # Ensure JSON contains a list of functions
                        data = [data]
                    all_data.extend(data)
            except FileNotFoundError:
                print(f"❌ File not found: {json_file}")
                continue

        # Insert all functions into MongoDB, avoiding duplicates
        inserted_count = 0
        for func in all_data:
            if "name" in func and "language" in func:
                self.insert_function(func)
                inserted_count += 1

        print(f"✅ Inserted {inserted_count} new function records into MongoDB.")

    def get_function_by_name(self, name: str):
        """
        Retrieves a function record from MongoDB by name.
        :param name: The name of the function to search for
        :return: The function details as a dictionary or a not found message
        """
        results = list(self.collection.find({"name": name}, {"_id": 0}))  # Exclude _id field
        return results if results else "❌ Function not found"

    def get_all_functions(self):
        """
        Retrieves all function records stored in the database.
        :return: A list containing all function details
        """
        return list(self.collection.find({}, {"_id": 0}))

    def delete_function_by_name(self, name: str):
        """
        Deletes a function record from MongoDB by name.
        :param name: The name of the function to delete
        """
        result = self.collection.delete_many({"name": name})
        if result.deleted_count:
            print(f"✅ Deleted {result.deleted_count} function(s): {name}")
        else:
            print(f"❌ Function not found: {name}")

    def clear_database(self):
        """
        Deletes all function records from MongoDB. (Use with caution in production)
        """
        self.collection.delete_many({})
        print("⚠️ All function data has been cleared from MongoDB")


# 📌 Test the database operations
if __name__ == "__main__":
    # Initialize the database
    db = FunctionDatabase()

    # Clear the database (optional)
    db.clear_database()

    # Import functions from Python & R JSON files
    json_files = ["py_functions.json", "R_function.json"]  # 也可以用 combined_functions.json
    db.insert_functions_from_json(json_files)

    # Retrieve all function records
    all_functions = db.get_all_functions()
    print(f"\n📌 The database contains {len(all_functions)} function records.")

    # Print only the first 5 function records
    for func in all_functions[:5]:
        print(json.dumps(func, indent=4, ensure_ascii=False))

    # Search for a specific function by name (it will return all matching languages)
    query_name = "RandomForestClassifier"
    result = db.get_function_by_name(query_name)
    print(f"\n🔍 Search result for '{query_name}':")
    print(json.dumps(result, indent=4, ensure_ascii=False))
