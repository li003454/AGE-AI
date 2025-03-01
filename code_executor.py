import subprocess
import os
import tempfile
import rpy2.robjects as robjects
from utils import *


@profile_func
def execute_python_code(code):
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
def execute_r_code(code):
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
