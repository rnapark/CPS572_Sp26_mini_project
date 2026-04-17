import re
from datasets import load_from_disk

ds = load_from_disk("opencode")

# filters

def has_python_function(text):
    return re.search(r"\bdef\s+\w+\s*\(", text) is not None


BAD_PATTERNS = [
    r"Here is (an|the)",
    r"Explanation",
    r"In this (code|function)",
    r"This function (does|will)",
    r"Let's",
    r"You can",
    r"To solve this",
    r"The following",
]

def is_explanatory(text):
    return any(re.search(p, text, re.IGNORECASE) for p in BAD_PATTERNS)


def extract_code_block(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


NON_PYTHON_HINTS = [
    r"<html",
    r"SELECT .* FROM",
    r"console\.log",
    r"function\s*\(",
    r"\bpublic\b|\bclass\b.*\{",
]

def is_non_python(text):
    return any(re.search(p, text, re.IGNORECASE) for p in NON_PYTHON_HINTS)


def is_reasonable_length(text):
    return 20 < len(text) < 1500

def has_substantive_logic(text):
    return (
        text.count("return") >= 1 and
        (text.count("if") + text.count("for") + text.count("while")) >= 1
    )

def has_return_or_logic(text):
    return (
        "return" in text or
        "for " in text or
        "if " in text
    )


def truncate_after_function(text):
    lines = text.split("\n")
    result = []
    indent_level = None

    for line in lines:
        if line.strip().startswith("def "):
            indent_level = len(line) - len(line.lstrip())

        if indent_level is not None:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent < indent_level:
                break

        result.append(line)

    return "\n".join(result)


# filter function

def filter_sample(sample):
    output = sample.get("output", "")
    
    if not output:
        return False

    # extract + clean
    output = extract_code_block(output)
    output = truncate_after_function(output)

    if not has_python_function(output):
        return False

    if is_non_python(output):
        return False

    if not is_reasonable_length(output):
        return False

    if not has_return_or_logic(output):
        return False

    if not has_substantive_logic(output):
        return False

    return True


# Apply filter

filtered_ds = ds.filter(
    filter_sample,
    num_proc=8  # parallelize (adjust to your CPU)
)

# Clean outputs

def clean_sample(sample):
    output = sample.get("output", "")
    output = extract_code_block(output)
    output = truncate_after_function(output)
    sample["output"] = output.strip()
    return sample


filtered_ds = filtered_ds.map(
    clean_sample,
    num_proc=8
)


final_pool = filtered_ds.select(range(min(10000, len(filtered_ds))))


final_pool.save_to_disk("opencode_filtered")

