#!/usr/bin/env python
# coding: utf-8
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2-2b-it"
model_path = f"./HF_Models/{model_id}"

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=model_path)


# ### 모델에 전달할 프롬프트를 임베딩합니다.
prompt = """너는 친절한 AI 지식 서비스 엔진이야. 
다음 질문에 답변해줘. 답변만 말해줘:  

질문: 대한민국 1월 평균 오전 날씨는 몇도야?
답변:"""
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

input_len = inputs.input_ids.shape[1]+ 10

outputs = model.generate(
    inputs.input_ids,
    do_sample=True,
    min_length=10,
    max_length=input_len+10,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3,
    temperature=1,
    top_k=50,
    top_p=0.92
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)