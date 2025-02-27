# Auto Generating and Executing AI Agent for Python/R

Auto Generating and Executing AI Agent for Python/R is a unified ai agent tool that extracts machine learning function metadata from both Python and R packages, stores them in a MongoDB database, indexes them with FAISS using OpenAI-generated embeddings, and provides an interactive interface to query, generate, and execute code based on user queries.

## Table of Contents

- [Installation and Dependency Setup](#installation-and-dependency-setup)
- [Example Queries and Expected Outputs](#example-queries-and-expected-outputs)
- [Expanding the Package Database](#expanding-the-package-database)
- [Usage](#usage)

## Installation and Dependency Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/li003454/AGE-ai.git
   cd AGE-ai
   ```

2. **Create a Virtual Environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Required Python Packages**

   The project depends on several libraries. You can install them using pip:

   ```bash
   pip install numpy openai faiss-cpu pymongo concurrent.futures rpy2
   ```

   *Note:* If you are using an operating system where FAISS is not available via pip, refer to the [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

4. **MongoDB Setup**

   - Make sure you have MongoDB installed and running.  
   - By default, the tool connects to MongoDB at `mongodb://localhost:27017/`. Adjust the URI if needed in the `Age` class constructor.

5. **Environment Variables**

   ## API Key Requirements

   This tool uses OpenAI's API for two main purposes:
   - **Code Generation:** It calls the GPT-4o model (a variant of GPT-4 optimized for coding tasks).
   - **Embeddings:** It utilizes the `text-embedding-3-small` model to generate embeddings for function records.

   **Important:**  
   Ensure your OpenAI API key has access to both GPT-4o and the text embedding models. 

   Set your OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"   # On Windows use: set OPENAI_API_KEY=your_openai_api_key
   ```
   - Note that you must have your own OPENAI_API_KEY to initialize the Age class, we don't provide a sample api key here(;

## Example Queries and Expected Outputs

Once the program is running, you will see a prompt in the interactive session. Below are some sample queries and what you might expect:(we also provide a main.py to help understand how to initialize the Age class, if you are confused, directly copy the main.py)

1. **Query Example:**

   **Input:**  
   ```
   Enter your query (or type 'exit' to quit): train a decision tree
   ```

   **Expected Output:**
   - The program will display a list of matched functions from the database, e.g.:
     ```
     🔍 **Matched Functions:**
     1. DecisionTreeClassifier (Package: sklearn.tree, Language: Python)
     2. DecisionTreeRegressor (Package: sklearn.tree, Language: Python)
     ```
   - If multiple functions are found, you will be prompted to select one.
   - After selection, the tool will generate a parameter list using OpenAI and prompt you to generate instance code.
   - Once confirmed, the generated Python script is displayed and executed, with the execution output (such as predictions on the Iris dataset) printed.

2. **Query Example for R Function:**

   **Input:**  
   ```
   Enter your query (or type 'exit' to quit): run k-means clustering in R
   ```

   **Expected Output:**
   - The program returns matched R functions, such as one from the `stats` package.
   - After selecting the appropriate function, the parameter list is filled, and the tool generates R code.
   - The generated R code is executed via rpy2, and the output (e.g., cluster assignments) is printed.

## Expanding the Package Database

To expand the list of packages from which function metadata is extracted:

1. **For Python Modules:**
   - Locate the `python_modules` list in the `update_database` method of the `Age` class.
   - Add the fully qualified module names you wish to include. For example:
     ```python
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
         "sklearn.feature_extraction.text",
         "your.new.module"  # add your new module here
     ]
     ```

2. **For R Packages:**
   - Locate the `r_packages` list in the `update_database` method of the `Age` class.
   - Add the names of the new R packages you want to include. For example:
     ```python
     r_packages = ["e1071", "caret", "MASS", "stats", "randomForest", "cluster", "yourNewRPackage"]
     ```
   - Make sure that the R package is installed in your R environment so that `rpy2` can successfully import it.

3. **Re-run the Program:**
   - After updating the lists, restart the program to re-extract and update the database with functions from the new packages.

## Usage

1. **Start the Program:**

   Run the script:
   ```bash
   python your_script_name.py
   ```

   If you're not still sure how to use it, run
   ```bash
   python main.py
   ```
   instead
   
   ### Make sure the age.py and your .py are in the same directory!

3. **Interactive Query Session:**

   - After initialization, the program enters an interactive session.
   - You can continuously enter queries and process function matches, generate code, and execute it.
   - To exit the session, type `exit` at the prompt.

4. **Examine Output:**

   - The program prints matched functions, generated parameter descriptions, generated code, and execution outputs directly in the console.

