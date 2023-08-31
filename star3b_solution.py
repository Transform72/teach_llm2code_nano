import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

checkpoint = "bigcode/starcoderbase-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float16)


def gen_model_solution(input_text):
    device = 'cuda'
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=256,
                             generation_config=GenerationConfig(
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
    ))
    return tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)


def gen_prompt(i):
    with open(f"testset/q{i}/prompt.txt", "r") as f:
        prompt = f.read()
    return prompt


def reformat_prompt(prompt):
    prompt = "<fim_prefix>\n" + prompt + "\n<fim_middle>"
    return prompt.replace("<FILL_ME>", "<fim_suffix>")


def gen_test_code(i):
    with open(f"testset/q{i}/test_code.py", "r") as f:
        test_code = f.read()
    return test_code

N = 3
accuracy = 0
for i in trange(N):
    prompt = gen_prompt(i)
    solution = gen_model_solution(reformat_prompt(prompt))
    test_code = gen_test_code(i)
    func2eval = prompt.replace("<FILL_ME>", solution)
    exec(test_code)
    accuracy += run_tests(func2eval)

print("Accuracy:")
print(accuracy / N)
