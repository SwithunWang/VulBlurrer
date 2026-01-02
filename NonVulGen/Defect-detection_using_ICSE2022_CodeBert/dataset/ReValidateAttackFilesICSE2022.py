import os
import pandas as pd

def traverse_files(path):
    path_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("attack_genetic_test_subs_"):
                file_path = os.path.join(root, file)
                path_list.append(file_path)
                # print(file_path)
    return path_list
# 调用函数遍历指定路径下的文件
path_list=traverse_files(
    "../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA")
print(path_list)
# exit()
# 初始化一个空的DataFrame
merged_df = pd.DataFrame()
for file in path_list:
    new_df=pd.read_csv(file)
    # 合并新的DataFrame到初始的DataFrame中
    merged_df = pd.concat([merged_df, new_df], ignore_index=True)
# 使用条件筛选，将符合条件的数据筛选出来
# 这里筛选出来的是所有含有漏洞的代码段
filtered_df_vul = merged_df[merged_df["True Label"] == 1]
print("测试数据集中含有漏洞的代码段样例个数为：",len(filtered_df_vul))
# 接下来筛选出来原来的模型预测准确的有漏洞代码段有多少个
filtered_df_vul_success = filtered_df_vul[filtered_df_vul["Original Prediction"] == 1]
print("测试数据集中被正确预测出含有漏洞的代码段样例个数为：",len(filtered_df_vul_success))
# 接下来我们计算这些被正确预测的代码段中被攻击成功的有几个
filtered_df_vul_success_robust = filtered_df_vul_success[filtered_df_vul_success["Is Success"] == 1]
print("测试数据集中被正确预测出含有漏洞的代码段，被成功鲁棒性攻击成功的个数为：",len(filtered_df_vul_success_robust),
      "攻击成功的比例为：",len(filtered_df_vul_success_robust)/len(filtered_df_vul_success))

# ['../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_0_400.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_1200_1600.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_1600_2000.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_2000_2400.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_2400_2800.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_400_800.csv',
# '../../attack-pretrain-models-of-code-ICSE2022/dataset_and_results/CodeBERT/Vulnerability Detection/data/attack results/GA\\attack_genetic_test_subs_800_1200.csv']
# 测试数据集中含有漏洞的代码段样例个数为： 1255
# 测试数据集中被正确预测出含有漏洞的代码段样例个数为： 491
# 测试数据集中被正确预测出含有漏洞的代码段，被成功鲁棒性攻击成功的个数为： 348 攻击成功的比例为： 0.7087576374745418，这里面验证的是yangzhou的做法，也就是攻击成功率为70.87%

