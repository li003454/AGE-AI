import query  # Importing the module we just created

# Initialize search engine and code generator
search_engine = query.SemanticSearch()
code_generator = query.CodeGenerator()

# User's query for an ML function
query_text = "Train a logistic regression model"
function_data = search_engine.search(query_text)

if function_data:
    print(f"\n🔍 Matched function: {function_data['name']} ({function_data['package']})")

    # Generate Python code using OpenAI API
    python_code = code_generator.generate_code_with_openai(function_data)
    print("\n📌 Generated Python Code:\n")
    print(python_code)

    # Execute the generated code
    print("\n🚀 Executing Code...\n")
    result = code_generator.execute_python_code(python_code)
    print(result)
else:
    print("❌ No matching function found.")

