__all__ = ["equivalence_prompt", "result_patterns"]


def equivalence_prompt(expr1: str, expr2: str):
    return f"""
You are an experienced mathematician. Consider the concept of “equivalence” between two mathematical
expressions or equations, meaning they share the same value or meaning and can be substituted for each
other without altering the outcome of a larger equation.

Examine the following expressions:
  1. {expr1}
  2. {expr2}

Determine whether they are equivalent. Return your answer in structured JSON format with exactly one key-value
pair where the key is `"result"` and the value is one of the following strings:
- `"YES"` if they are equivalent
- `"NO"` if they are not
- `"DONTKNOW"` if you cannot decide
"""


result_patterns = [r'json\{"result":"(.*?)"\}', r'\{"result":"(.*?)"\}']
