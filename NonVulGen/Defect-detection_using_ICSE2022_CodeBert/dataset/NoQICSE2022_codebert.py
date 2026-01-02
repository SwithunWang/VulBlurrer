import os
import re
import sys

import pandas as pd
sys.path.append('../code')
from run_robustness_attackV2 import main
def traverse_files(path):
    path_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("attack_gi_"):
                file_path = os.path.join(root, file)
                path_list.append(file_path)
                # print(file_path)
    return path_list
# 调用函数遍历指定路径下的文件
path_list=traverse_files(
    "../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA")
print(path_list)

import re
dict_noq={}
for path in path_list:
    print(path)
    index1=path.index('attack_gi_test_subs_')
    # print(index1)
    index2 = path.rindex('_')
    # print(index2)
    # print(path[index1+len('attack_gi_test_subs_'):index2])
    index_base=int(path[index1+len('attack_gi_test_subs_'):index2])
    # 循环读取path的每行数据
    # Query times in this attack

    cur_index=0
    # 打开文件
    with open(path, "r") as file:
        # 逐行读取文件
        for line in file:
            # 查找包含特定文本的行
            if "Query times in this attack" in line:
                # 使用正则表达式提取数字
                match = re.search(r'\d+', line)

                if match:
                    number = int(match.group())
                    # print("Found number:", number)
                    dict_noq[index_base + cur_index] = number
                    cur_index += 1
                else:
                    print("No number found in this line")

print(dict_noq)

results = main(test_data_file="test.jsonl", testing=True, isGenerateTestSuccessSamples=False,
               isRevalidate=False)
print(results['labels'])
print(results['preds'])
print(len(results['preds']))
# 便利列表，把lable=1，与preds为True的样例取出来，计算攻击的次数
all_attack_times=0
attack_times=0
for i in range(len(results['preds'])):
    if results['labels'][i]==1 and  results['preds'][i]==True:
        attack_times+=1
        all_attack_times+=dict_noq[i]
print("成功攻击的样例为：",attack_times)
print("平均攻击次数为：", all_attack_times/attack_times)
# 成功攻击的样例为： 491
# 平均攻击次数为： 566.8533604887983



