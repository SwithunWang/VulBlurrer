# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
js_all = json.load(open('function.json'))
train_index = set()
valid_index = set()
test_index = set()

with open('train.txt') as f:
    for line in f:
        line = line.strip()
        train_index.add(int(line))

with open('valid.txt') as f:
    for line in f:
        line = line.strip()
        valid_index.add(int(line))

with open('test.txt') as f:
    for line in f:
        line = line.strip()
        test_index.add(int(line))

def dataset_generation():
    examples = 0
    # 计算样例的长度
    # 以下是操作生成训练数据集。
    with open('train.jsonl', 'w') as f:
        for idx, js in enumerate(js_all):
            if idx in train_index:
                js['idx'] = idx
                examples = examples + 1
                f.write(json.dumps(js) + '\n')
        #接下来输出新生成的训练数据集
        js_new_attaks=json.load(open('AttackResluts.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
        js_new_attaks = json.load(open('AttackResluts2.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
        js_new_attaks = json.load(open('AttackResluts3.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
        js_new_attaks = json.load(open('AttackResluts4.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
        js_new_attaks = json.load(open('AttackResluts5.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
        js_new_attaks = json.load(open('AttackResluts6.jsonl'))
        for idx, js in enumerate(js_new_attaks):
            examples = examples + 1
            f.write(json.dumps(js) + '\n')
    print("训练数据集长度为：", examples)
     # 以下是操作生成验证数据集。
    with open('valid.jsonl', 'w') as f:
        for idx, js in enumerate(js_all):
            if idx in valid_index:
                js['idx'] = idx
                f.write(json.dumps(js) + '\n')
    # 以下是操作生成测试数据集。
    with open('test.jsonl', 'w') as f:
        for idx, js in enumerate(js_all):
            if idx in test_index:
                js['idx'] = idx
                f.write(json.dumps(js) + '\n')

# 记录开始运行时间
import datetime
start = datetime.datetime.now()
dataset_generation()
# 记录程序结束时间
end = datetime.datetime.now()
print(("此次生成数据集共花费的时间为：%s", str(end - start)))