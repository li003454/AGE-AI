import openai
import re
from utils import *


@profile_func
def fill_parameters(api_key, function_data):
    """
    Call the OpenAI API to generate a complete parameter list based on the function record.
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
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a parameter assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    filled_parameters = response.choices[0].message.content.strip()
    return filled_parameters

@profile_func
def generate_code(api_key, function_data):
    """
    Call the OpenAI API to generate complete code in the corresponding language (Python or R)
    based on the function record.
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
    else:
        return None

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
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
