import os

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # 禁用 GPU

import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 统一路径分隔符，使用原始字符串 r'path'
tokenizer = AutoTokenizer.from_pretrained(r"C:\pyprojects\lessons\NLP\qa_model")
model_trained = AutoModelForSeq2SeqLM.from_pretrained(r"C:\pyprojects\lessons\NLP\qa_model")

# 检查 GPU 可用性
device = 'cuda' if cuda.is_available() else 'cpu'
model_trained.to(device)


# 预处理和后处理函数
def preprocess(text):
  """将输入中的换行符替换为下划线"""
  return text.replace("\n", "_")


def postprocess(text):
  """去除特定的符号"""
  return text.replace(".", "").replace('</>', '')


def answer_fn(text, sample=False, top_k=50):
  '''生成答案的函数
  参数:
  - text: 输入的问题
  - sample: 是否使用采样生成答案
  - top_k: 生成时的top-k候选选择
  '''
  encoding = tokenizer(
    text=[text],
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
  ).to(device)

  if not sample:  # 不进行采样
    out = model_trained.generate(
      **encoding,
      return_dict_in_generate=True,
      max_length=512,
      num_beams=4,
      temperature=0.5,
      repetition_penalty=10.0,
      remove_invalid_values=True
    )
  else:  # 使用采样
    out = model_trained.generate(
      **encoding,
      return_dict_in_generate=True,
      max_length=512,
      temperature=0.6,
      do_sample=True,
      repetition_penalty=3.0,
      top_k=top_k
    )

  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  if out_text[0] == '':
    return '我只是个语言模型，这个问题我回答不了。'
  return postprocess(out_text[0])


# 与用户的交互循环
text_list = []
while True:
  text = input('请输入问题:')
  if text.lower() == 'exit':  # 输入 'exit' 退出循环
    break
  result = answer_fn(text, sample=True, top_k=100)
  print("模型生成:", result)
  print('*' * 100)



      