from __future__ import absolute_import, division, print_function

import multiprocessing

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter
cpu_cont = multiprocessing.cpu_count()
import transformers
# from transformers import (get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
import torch
import sklearn
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.__version__)
print(transformers.__version__)
print(sklearn.__version__)
exit()
import random
import time

counter = 0  # 定义一个计数器，用于增加随机性

def generate_random_vector():
    global counter  # 在函数中使用全局变量
    counter += 1  # 计数器加1
    random.seed(str(time.time()) + str(counter))  # 使用当前时间和计数器的值作为随机数种子
    vector = [0] * 15  # 初始化一个长度为15的向量，所有位置的值都为0
    ones_indices = random.sample(range(15), 8)  # 随机选取5个位置，将其值设为1
    for index in ones_indices:
        vector[index] = 1
    return vector

for i in range(100):
    # 调用函数生成向量
    random_vector1 = generate_random_vector()
    random_vector2 = generate_random_vector()
    print(random_vector1)
    print(random_vector2)