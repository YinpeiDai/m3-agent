# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import sys
import json
import time
import openai
import argparse
import multiprocessing
import mmagent.videograph
from mmagent.retrieve import search
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages
from mmagent.prompts import prompt_agent_verify_answer_referencing
from tqdm import tqdm

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
model_name = "models/M3-Agent-Control"
config = json.load(open("configs/api_config.json"))
gpt_model = "gpt-4-0125-preview"
client = openai.AzureOpenAI(
    azure_endpoint=config[gpt_model]["azure_endpoint"],
    api_version=config[gpt_model]["api_version"],
    api_key=config[gpt_model]["api_key"],
)

def get_response(messages, timeout=30):
    response = client.chat.completions.create(
        model=gpt_model, messages=messages, temperature=0, timeout=timeout, max_tokens=2048
    )
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(messages, timeout=30):
    for i in range(20):
        try:
            return get_response(messages, timeout)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after 5 retries")

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=ground_truth,
                    agent_answer=predict,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(messages)
        result = response[0].lower()
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return False
    return True if "yes" in result else False

system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""

tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=1024
)
pattern = r"Action: \[(.*)\].*Content: (.*)"

def consumer(data):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)
        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None
        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                mem_node.refresh_equivalences()
                if "character id" in content:
                    memories, _, _ = search(mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip)
                    new_memories.update(memories)
                else:
                    memories, currenr_clips, _ = search(mem_node, content, data["currenr_clips"], threshold=0.5, topk=processing_config["topk"], before_clip=before_clip)
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
            search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")
    args = parser.parse_args()
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    output_path = os.path.join("data/results", f"{dataset_name}.jsonl")
    model = LLM(model=model_name, tensor_parallel_size=2, dtype="bfloat16", gpu_memory_utilization=0.95)

    batched_datas, data = [], []
    datas = json.load(open(args.data_file))
    for _, v in datas.items():
        for qa in v["qa_list"]:
            data.append({
                "id": qa["question_id"],
                "mem_path": v["mem_path"],
                "question": qa["question"],
                "answer": qa["answer"],
            })
            if "before_clip" in qa:
                data[-1]["before_clip"] = qa["before_clip"]
            if len(data) == processing_config["batch_size"]:
                batched_datas.append(data)
                data = []
    if len(data) > 0:
        batched_datas.append(data)

    result = []
    for batched_data in tqdm(batched_datas[:3]):
        for i in range(len(batched_data)):
            batched_data[i]["conversations"] = [{"role": "system", "content": system_prompt.format(question=batched_data[i]["question"])}, {"role": "user", "content": "Searched knowledge: {}"}]
            batched_data[i]["finish"] = False
            batched_data[i]["currenr_clips"] = []

        for idx in range(processing_config["total_round"]):
            vllm_inputs = []
            for data in batched_data:
                if data["finish"]:
                    continue
                data["conversations"][-1]["content"] += instruction
                if idx == processing_config["total_round"] - 1:
                    data["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                text = tokenizer.apply_chat_template(
                    data["conversations"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                vllm_inputs.append({"prompt_token_ids": text})

            outputs = model.generate(
                prompts=vllm_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            i = 0
            for data in batched_data:
                if data["finish"]:
                    continue
                data["conversations"].append({"role": "assistant", "content": outputs[i].outputs[0].text})
                i += 1
            assert i == len(vllm_inputs)
            
            with multiprocessing.Pool() as pool:
                batched_data = pool.map(consumer, batched_data)

        for data in batched_data:
            if "response" in data:
                data["gpt_eval"] = eval_answer(data["question"], data["response"], data["answer"])
                time.sleep(0.5)
            else:
                data["gpt_eval"] = False
            result.append(json.dumps(data, ensure_ascii=False) + '\n')

    with open(output_path, "w") as f:
        for i in result:
            f.write(i)
