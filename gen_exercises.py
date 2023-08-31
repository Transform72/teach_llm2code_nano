import pickle, os, re, time, json
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# Load the environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
OPENAI_KEY = os.getenv('OPENAI_KEY')

# Set the API key for OpenAI
openai.api_key = OPENAI_KEY


def gpt_completion(prompt, n, model, temperature=1, max_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    )
    raw_responses = [response.choices[i].message["content"] for i in range(n)]
    return raw_responses


def safe_gpt_completion(prompt, n, model="gpt-3.5-turbo", max_retries=20):
    retries = 0
    while retries < max_retries:
        try:
            return gpt_completion(prompt, n, model)
        except (openai.error.RateLimitError, openai.error.APIConnectionError, openai.error.APIError, openai.error.Timeout) as e:
            print(type(e), e)
            time.sleep(60)
            retries += 1
    print(f"Failed after {max_retries} attempts.")
    return None


def single_concept_exercise_prompt(concept):
    prompt = f'''You are a data science tutor and you are designing exercises for python Pandas library. The concept you want to teach is: 
{concept}

The exercise should have the following format:
1. The problem is a python function with very detailed description of what it is supposed to do and there should be a chunk of code missing in the function indicated by "<FILL_ME>" that needs to be filled in by the students. 
2. The solution should be the code that replaces the "<FILL_ME>" that fully completes the function.
3. It is extremely important that when we replace the "<FILL_ME>" with the solution code, the function should return the correct result.

Here is an example:

PROBLEM:
```python
def add_function(df, col1, col2):
    """
    Create a new DataFrame with an additional column 'sum', which contains the sum of col1 and col2

    Parameters:
    - df (DataFrame): Original DataFrame
    - col1 (str): Name of the first column to consider
    - col2 (str): Name of the second column to consider

    Returns:
    - DataFrame: New DataFrame with an additional 'sum' column.

    Example:
    ---------
    Input DataFrame:
    ----------------
        col1  col2  col3
    0     1    2    3

    Function call:
    --------------
    get_new_df_1(df, 'col2', 'col3')

    Output DataFrame:
    -----------------
        col1  col2  col3   sum
    0      1   2    3      5
    """
<FILL_ME>
    return new_df
```

SOLUTION:
```python
    new_df = df.copy()
    new_df['sum'] = new_df['col1'] + new_df['col2']
```

Please redpond with the same format as the example above.
'''
    return prompt


def get_concepts():
    return ["Column Access and Manipulation: Familiarity with selecting DataFrame columns by their names and applying operations to them.",
            "Conditional Logic: Understanding how to create and apply conditions on DataFrame columns. This usually involves comparison operators and bitwise operators (& for 'and', | for 'or').",
            "Lambda Functions and apply Method: Using the apply method along with lambda functions to perform row-wise operations based on conditions, especially when the conditions involve multiple columns.",
            "DataFrame Method Chaining: Combining multiple DataFrame methods or operations in a sequence (chaining).",
            "Column Creation: Creating new columns in a DataFrame based on existing columns and conditions."]


def gen_single_concept_exercises():
    concept_list = get_concepts()
    exercises = []
    for concept in tqdm(concept_list):
        prompt = single_concept_exercise_prompt(concept)
        gpt_responses = safe_gpt_completion(prompt, 10)
        for r in gpt_responses:
            exercises.append(r)

    with open("single_concept_exercises.pk", "wb") as f:
        pickle.dump(exercises, f)


if __name__ == "__main__":
    gen_single_concept_exercises()
