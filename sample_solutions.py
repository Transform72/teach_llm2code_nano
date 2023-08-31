def gen_solution(i):
    with open(f"testset/q{i}/sample_solution.txt", "r") as f:
        solution = f.read()
    return solution


def gen_prompt(i):
    with open(f"testset/q{i}/prompt.txt", "r") as f:
        prompt = f.read()
    return prompt


def gen_test_code(i):
    with open(f"testset/q{i}/test_code.py", "r") as f:
        test_code = f.read()
    return test_code

N = 3
accuracy = 0
for i in range(N):
    solution = gen_solution(i)
    prompt = gen_prompt(i)
    test_code = gen_test_code(i)
    func2eval = prompt.replace("<FILL_ME>", solution)
    exec(test_code)
    accuracy += run_tests(func2eval)

print("Accuracy:")
print(accuracy / N)
