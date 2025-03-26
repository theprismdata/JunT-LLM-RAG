from openai import OpenAI

openai_api_key = "DUMMY"

openai_api_base = "http://192.168.205.174:8889/v1"
"""
meta-llama/Llama-3.2-3B-Instruct
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8889 --max-model-len=256
"""

"""
yanolja/EEVE-Korean-10.8B-v1.0
python -m vllm.entrypoints.openai.api_server --model yanolja/EEVE-Korean-10.8B-v1.0 --host 0.0.0.0 --port 8889 --max-model-len=256
"""
model_path = "yanolja"
model_code = "EEVE-Korean-10.8B-v1.0"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

result = client.completions.create(model = f'{model_path}/{model_code}',
                          prompt = "강남의 맛집을 찾아줘")
print(result.choices[0].text)