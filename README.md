# teach_llm2code_nano
code for blog post "Teaching Large Language Models to Master Code: A Nano-Scale Example"

## Test Cases
3 manually created test examples can be found in `testset`, with each having prompt, sample solution and test code.

They are all tests of modells' understanding of Pandas column manupulation, conditional operators and apply method.

Checkout `sample_solutions.py` to see how tests can be performed (which has accuracy: 1.0).

## Base Model
[starcoderbase-3b](https://huggingface.co/bigcode/starcoderbase-3b) is used as the base model.

If we run `star3b_solution.py`, we will see that it gets an accuracy of 0.33, which means it fails 2 out of 3 
test cases. 

## Data Preparation
**Note: This is the most important step.**

50 examples are generated with `gpt-3.5-turbo` with `gen_exercises.py` with seed concepts as part of the prompt.

And then we need to manually inspect the generated code and remove duplicated examples as well as fixing errors, as shown in
`human-in-the-loop single concept training data cleaning.ipynb`. 26 examples are kept and saved in `single_concept_train.json`.

## Fine Tune
Run `fine_tune_star3b.py` to fine tune the base model with the 26 examples generated.

And then run `fine_tuned_solution.py` to evaluate the model. And it should print `Accuracy: 1.0`.