import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import transformers
from transformers import AutoTokenizer

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    # command = "Give your answer within 128 tokens."
    import random
    # numbers = [32,64,128,256,512]
    # problem_prompt = "You are a helpful assistant."
    # 读取原始JSON文件
    with open(data_path, 'r') as input_file:
        metamath = json.load(input_file)
    # print('promt =====', problem_prompt)
    for item in metamath:
            temp_instr = item["instruction"]
            gsm8k_ins.append(temp_instr)
            temp_ans = item['output']
            # temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512)
    # print('sampleing =====', sampling_params)
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    data = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        result.append(False)
        input_token = tokenizer.tokenize(prompt)
        output_token = tokenizer.tokenize(completion)
        temp = {'question': prompt, 'output': completion, 'input_text_length': len(prompt), 'input_token_length': len(input_token), 'outut_text_length': len(completion),'output_token_length': len(output_token)}
        data.append(temp)

    # Save the data into a JSON dataset
    dataset_path = "/mnt/octave/data/siqizhu/ActivationDirectionAnalysis/alpaca_output.json"
    with open(dataset_path, 'a') as file:
        json.dump(data, file, indent=4)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    # parser.add_argument("--target_model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=4000)  # end index
    parser.add_argument("--batch_size", type=int, default=800)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
