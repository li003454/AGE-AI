import json
import re
import importlib
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from utils import *

# ─── Python Function Extraction ─────────────────────────────────────────────

@profile_func
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

@profile_func
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

@profile_func
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

@profile_func
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

@profile_func
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