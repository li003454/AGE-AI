from query import SemanticSearch, CodeGenerator

# 初始化搜索引擎和代码生成器
search_engine = SemanticSearch()
code_generator = CodeGenerator()

# 用户查询
query_text = "Train a logistic regression model"
top_k = 3  # 允许返回多个匹配项

# 进行搜索
search_results = search_engine.search(query_text, top_k=top_k)

if search_results and isinstance(search_results, list):
    print("\n🔍 **Matched Functions:**")

    # 显示所有匹配的函数
    for i, func in enumerate(search_results):
        print(f"{i+1}. {func['name']} (Package: {func['package']}, Language: {func['language']})")

    # 如果有多个匹配项，提示用户选择
    if len(search_results) > 1:
        choice = input("\n👉 Enter the number of the function you want to use: ")
        try:
            selected_func = search_results[int(choice) - 1]
        except (IndexError, ValueError):
            print("❌ Invalid selection. Exiting...")
            exit(1)
    else:
        selected_func = search_results[0]  # 只有一个匹配项时直接使用

    print(f"\n✅ Selected: {selected_func['name']} ({selected_func['language']})")

    # 生成代码
    code = code_generator.generate_code_with_openai(selected_func)
    print("\n📌 Generated Code:\n", code)

    # 根据语言执行代码
    if selected_func["language"].lower() == "python":
        print("\n🚀 Executing Python Code...\n")
        result = code_generator.execute_python_code(code)
    elif selected_func["language"].lower() == "r":
        print("\n🚀 Executing R Code...\n")
        result = code_generator.execute_r_code(code)
    else:
        print(f"❌ Unsupported language: {selected_func['language']}")
        result = None

    if result:
        print("\n📊 Execution Output:\n", result)
else:
    print("❌ No matching function found.")
