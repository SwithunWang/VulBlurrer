import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import random
import math
# from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5EncoderModel
from mutation import TransformCodePreservingSemantic_orient
from test_escape_score import (
    compute_similarity,get_codebleu,compute_fluency
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

codebleu_dir = os.path.join(PROJECT_ROOT, "codebleu-0.6.0")
sys.path.append(codebleu_dir)
from codebleu import calc_codebleu

code_dir = os.path.join(PROJECT_ROOT, "code")
sys.path.append(code_dir)
import run_robustness_attackV2 

# 指定日志目录
log_dir = os.path.join(PROJECT_ROOT, "log_records")
os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在就创建
sys.path.append(log_dir)

global_processes=5
global_cur_process=0
global_code_count=15
global_mutationCount=8
global_num_generation=10
global_population_size=6
global_alpha_0=0.4
global_beta_0=0.35
global_gamma_0=0.25

global_memo='test_codebert_vulblurrer'

# 构造日志文件名
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_name = f"{script_name}_pro_{global_processes}_cur_{global_cur_process}_memo_{global_memo}.log"
log_path = os.path.join(log_dir, log_name)

# 创建 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 文件 handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)

# 格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加 handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"日志保存到 {log_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 注意路径
codet5_model_path = "../models/codet5"
codebert_model_path = "../models/codebert"

embed_tokenizer = AutoTokenizer.from_pretrained(codet5_model_path, local_files_only=True)
embed_model = AutoModel.from_pretrained(codet5_model_path, local_files_only=True).to(device)
embed_model.eval()

mlm_tokenizer = AutoTokenizer.from_pretrained(codebert_model_path, local_files_only=True)
mlm_model = AutoModelForMaskedLM.from_pretrained(codebert_model_path, local_files_only=True).to(device)
mlm_model.eval()

# 定义染色体长度
chromosome_length=15

# 交叉
def crossover(p1_chromosome, p2_chromosome):
    # 选择一个交叉点
    crossover_point = random.randint(1, chromosome_length - 1)
    # 生成两个子代
    child1 = p1_chromosome[:crossover_point] + p2_chromosome[crossover_point:]
    child2 = p2_chromosome[:crossover_point] + p1_chromosome[crossover_point:]
    return child1, child2

# 变异
def mutation(chromosome):
    # 选择一个变异位点
    mutation_point = random.randint(0, chromosome_length - 1)
    # 变异位点取反
    chromosome[mutation_point] = 1 - chromosome[mutation_point]
    return chromosome

def tournament_selection(population, tournament_size=3):
    candidates = random.sample(population, min(tournament_size, len(population)))
    candidates.sort(key=lambda x: x['esc_score'], reverse=True)
    return candidates[0]

def update_weights_dynamic(iteration, alpha0=0.4, beta0=0.35, gamma0=0.25, lambda_val=0.05):
    alpha_i = alpha0 + (1 - alpha0) * (1 - math.exp(-lambda_val * iteration))
    remain = 1.0 - alpha_i
    norm = beta0 + gamma0

    beta_i = remain * (beta0 / norm)
    gamma_i = remain * (gamma0 / norm)

    return alpha_i, beta_i, gamma_i

counter = 0

def operator_vector_weighted(mutationCount=8):
    # 优先对 if、method、for 相关语句进行扰动（概率倾斜）
    global counter
    counter += 1
    random.seed(str(time.time()) + str(counter))

    vector = [0] * chromosome_length

    priority_ops = [2, 4, 5, 6, 13]  # 优先扰动算子
    priority_indices = [i-1 for i in priority_ops]

    weights = []
    for i in range(chromosome_length):
        if i in priority_indices:
            weights.append(5.0)
        else:
            weights.append(1.0)

    # 根据权重随机采样
    selected_indices = random.choices(population=list(range(chromosome_length)), weights=weights, k=mutationCount*2)
    selected_indices = list(set(selected_indices))[:mutationCount]
    for idx in selected_indices:
        vector[idx] = 1

    return vector

def escape_score(ori_code, adv_code):
    
    # 1.特征一致性
    sim = compute_similarity(ori_code, adv_code,embed_tokenizer, embed_model, device)

    # 2.CodeBLEU
    codebleu = get_codebleu(ori_code, adv_code)

    # 3.FLU
    flu = compute_fluency(adv_code, mlm_tokenizer, mlm_model, device)

    # score = alpha * sim_norm + beta * codebleu + gamma * flu_norm
    return {
            "sim": sim, 
            "codebleu": codebleu,
            "flu": flu
        }
        
def eval_escape_score(vector,code,target,code_idx):
    '''
    # 定义逃逸分数函数
    # vector: 代码同义转换操作向量
    # code: 原始代码
    # target: 原始代码的ground-true label
    # code_idx: 原始代码的索引
    '''
    TRANSFORMCODE = f"./current_code_for_esc_socre_{global_cur_process}.c"
    with open(TRANSFORMCODE, "w") as file:
        file.write(code)
    content = None
    for index in range(1, len(vector) + 1):
        if vector[index - 1] == 0:
            continue
        elif vector[index - 1] == 1:
            ACTION = index
            content = TransformCodePreservingSemantic_orient(TRANSFORMCODE=TRANSFORMCODE, ATCION=ACTION,code=code,cur_process=global_cur_process)
            if content == None:
                continue
            with open(TRANSFORMCODE, "wb") as file:
                file.write(content)
    if content == None:
        return None,None

    data = {}
    data['project'] = 'test'
    data['commit_id'] = 'test'
    data['target'] = target
    # 这里把二进制格式的字符串转换为常规字符串，要注意是不是会有异常出现。
    data['func'] = content.decode('utf-8')
    data['idx'] = code_idx

    ori_code = code
    adv_code = data['func']

    result = escape_score(ori_code, adv_code)
    
    return data,result

def genetic_algorithm_new(ori_code,
                      target,
                      init_population,
                      code_idx,
                      num_generations=10,
                      elite_size=5,
                      tournament_size=3,
                      mutation_rate=0.1,
                      child_count=4):
    population = []
    for s in init_population:
        population.append({
            "vector": s["vector"],
            "code": s["data"],
            "esc_score_dict": s["esc_score_dict"],  # 原始指标
            "esc_score": s["esc_score"],   # Top-K 阶段算好的
        })

    best_adv_code = None

    for i in range(num_generations):
        if i == 0:
            population.sort(key=lambda x: x['esc_score'], reverse=True)
            parent1 = population[0]
            parent2 = population[1]
        else:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
        
        child_population = []
        loops = (child_count + 1) // 2

        for _ in range(loops):
            # 交叉
            c1_vec, c2_vec = crossover(parent1['vector'], parent2['vector'])
            # 变异
            if random.random() < mutation_rate:
                c1_vec = mutation(c1_vec)
            if random.random() < mutation_rate:
                c2_vec = mutation(c2_vec)

            for vec in [c1_vec, c2_vec]:
                data, esc_score_dict = eval_escape_score(vector=vec, code=ori_code, target=target, code_idx=code_idx)
                if data is None or esc_score_dict is None:
                    continue
                # 保存到 child_population
                child_sample = {
                    'vector': vec,
                    'data': data,
                    'esc_score_dict': esc_score_dict,
                    'esc_score': 0.0
                }
                child_population.append(child_sample)
                logger.info(f"生成第{len(child_population)}个子代, vector={vec}, esc_score={esc_score_dict}")

                # 直接查询模型
                adv_code = data
                adv_path = PROJECT_ROOT + f"../check_child_{global_cur_process}.jsonl"
                with open(adv_path, 'w') as f:
                    f.write(json.dumps(adv_code) + '\n')
                global noq
                results = run_robustness_attackV2.main(test_data_file=adv_path, testing=True, isGenerateTestSuccessSamples=False, isRevalidate=False, target_model_path=target_model_path)
                noq += 1
                labels = results['labels']
                preds = results['preds']
                logits = results['logits']
                checkresult = None
                certaintyValue = None
                if len(labels) == 1 and len(preds) == 1:
                    tmplabels=labels[0]
                    tmppreds=preds[0]
                    int_tmp_preds = int(tmppreds)
                    checkresult = int_tmp_preds != tmplabels
                if len(logits) == 1:
                    certaintyValue = logits[0, 0]
                if checkresult is not None and certaintyValue is not None and adv_code is not None:
                    if checkresult:
                        best_adv_code = adv_code
                        logger.info(f"[GA] 第 {i} 轮攻击成功")
                        return best_adv_code
        child_population = child_population[:child_count]

        alpha_i, beta_i, gamma_i = update_weights_dynamic(i)

        merged_population = population + child_population

        sim_vals = [s["esc_score_dict"]["sim"] for s in merged_population]
        cb_vals = [s["esc_score_dict"]["codebleu"] for s in merged_population]
        flu_vals = [s["esc_score_dict"]["flu"] for s in merged_population]

        min_sim, max_sim = min(sim_vals), max(sim_vals)
        min_cb, max_cb = min(cb_vals), max(cb_vals)
        min_flu, max_flu = min(flu_vals), max(flu_vals)

        def min_max_norm(x, min_v, max_v):
            return 0.0 if max_v == min_v else (x - min_v) / (max_v - min_v)

        for s in merged_population:
            sim_n = min_max_norm(s["esc_score_dict"]["sim"], min_sim, max_sim)
            cb_n = min_max_norm(s["esc_score_dict"]["codebleu"], min_cb, max_cb)
            flu_n = min_max_norm(s["esc_score_dict"]["flu"], min_flu, max_flu)

            s["esc_score"] = (alpha_i * sim_n + beta_i * cb_n + gamma_i * flu_n)
    

        merged_population.sort(key=lambda x: x["esc_score"], reverse=True)
        population = merged_population[:elite_size]

        logger.info(
            f"[GA] 第 {i} 轮结束 | best esc={population[0]['esc_score']:.4f} "
            f"| alpha={alpha_i:.3f}, beta={beta_i:.3f}, gamma={gamma_i:.3f}"
        )

    logger.info("[GA] Attack failed after all generations")
    return None

def iterateMutation_new(test_file_path = 'test_success_examples.jsonl',
                    processes=6,  
                    cur_process=0,  #当前的进程号
                    population_size=6,  
                    num_generations=10, 
                    mutationCount=8,
                    code_count=15,
                    ):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_DIR = PROJECT_ROOT + "/results/"
    success_results_path = RESULTS_DIR+ "AttackResluts_process_"+ str(processes)+ "_"+ str(cur_process)+ ".jsonl"
    success_results_path_with_ori_code = RESULTS_DIR+ "AttackResluts_ori_process_"+ str(processes)+ "_"+ str(cur_process)+ ".jsonl"
    logger.info(
        f"processes={processes}, cur_process={cur_process}, population_size={population_size}, "
        f"code_count={code_count}, num_generations={num_generations}, mutationCount={mutationCount}")
    logger.info("当前运行的进程保存的路径是："+success_results_path)

    success_results = []
    success_results_with_ori = []
    js_all = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            js_all.append(json.loads(line.strip()))
    count_processed = 0
    count_success_attack = 0
    
    for idx, js in enumerate(js_all):
        if js['target']==1:
            count_processed = count_processed + 1
            ori_code = js['func']
            code_idx = js['idx']
            logger.info("当前代码段的索引是：" + str(code_idx))   
            
            best_adv_code = None
            num_samples = code_count  # 生成对抗样本数量
            topk = 5          # top-k EscapeScore
            candidate_samples = []

            # 1: 生成对抗样本
            for i in range(num_samples):
                vector = operator_vector_weighted(mutationCount)
                data, esc_score_dict = eval_escape_score(
                    vector=vector,
                    code=ori_code,
                    target=js['target'],
                    code_idx=code_idx
                )
                if data is not None and esc_score_dict is not None:
                    candidate_samples.append({
                        'vector': vector,
                        'data': data,
                        'esc_score_dict': esc_score_dict,
                        'esc_score': 0.0
                    })
                    logger.info(f"生成第{i+1}个对抗样本, vector={vector}, esc_score={esc_score_dict}")
                else:
                    logger.warning("返回的数据为空，要小心了")
            if not candidate_samples:
                continue

            # 2: 归一化 + 计算逃逸分数 + 排序
            sim_vals = [s['esc_score_dict']['sim'] for s in candidate_samples]
            codebleu_vals = [s['esc_score_dict']['codebleu'] for s in candidate_samples]
            flu_vals = [s['esc_score_dict']['flu'] for s in candidate_samples]

            # min-max 归一化函数
            def min_max_norm(x, min_val, max_val):
                return 0.0 if max_val == min_val else (x - min_val) / (max_val - min_val)

            # 设置权重

            for s in candidate_samples:
                sim_n = min_max_norm(s['esc_score_dict']['sim'], min(sim_vals), max(sim_vals))
                codebleu_n = min_max_norm(s['esc_score_dict']['codebleu'], min(codebleu_vals), max(codebleu_vals))
                flu_n = min_max_norm(s['esc_score_dict']['flu'], min(flu_vals), max(flu_vals))

                # 综合 escape score
                s['esc_score'] = global_alpha_0 * sim_n + global_beta_0 * codebleu_n + global_gamma_0 * flu_n
            #排序      
            candidate_samples.sort(key=lambda x: x['esc_score'], reverse=True)

            # 3: 查询 top-k 对抗样本 
            attack_success = False
            for idx1, sample in enumerate(candidate_samples[:topk]):
                adv_code = sample['data']
                adv_path = PROJECT_ROOT + "/dataset/check_topk.jsonl"
                with open(adv_path, 'w') as f:
                    f.write(json.dumps(adv_code) + '\n')
                global noq
                results = run_robustness_attackV2.main(test_data_file=adv_path,testing=True, isGenerateTestSuccessSamples=False, isRevalidate=False,target_model_path=target_model_path)
                noq += 1
                labels=results['labels']
                preds=results['preds']
                logits= results['logits']
                if len(labels) == 1 and len(preds) == 1:
                    tmplabels=labels[0]
                    tmppreds=preds[0]
                    int_tmp_preds = int(tmppreds)
                    checkresult = int_tmp_preds != tmplabels
                    logger.info(f"[Top-K] idx={code_idx}, sample_rank={idx1}, attack_success={checkresult}")
                if len(logits) == 1:
                    certaintyValue=logits[0,0] 
                if data != None and checkresult != None and certaintyValue!=None:
                    if checkresult:
                        attack_success = True
                        count_success_attack += 1
                        best_adv_code = adv_code
                        logger.info(f"逃逸topk中攻击成功, best_adv_code 记录为: {best_adv_code}, idx={code_idx}")
                        break  # 找到第一个成功的就可以停止

            # 4: 遗传算法优化 
            if not attack_success:
                logger.info("Top-K failed, entering GA stage")
                init_population = candidate_samples[:topk]
                best_adv_code = genetic_algorithm_new(ori_code,js["target"],
                                                    init_population,code_idx,num_generations=num_generations,
                                                    elite_size=2,tournament_size=5,mutation_rate=0.2)

                if best_adv_code is not None:
                    attack_success = True
                    count_success_attack += 1

            if attack_success:
                success_results.append(best_adv_code)
                success_results_with_ori.append({
                    "idx": code_idx,
                    "ori_code": ori_code,
                    "adv_code": best_adv_code
                })

                with open(success_results_path, "w") as f:
                    json.dump(success_results, f)

                with open(success_results_path_with_ori_code, "w") as f:
                    json.dump(success_results_with_ori, f)

            logger.info("当前进程已处理的代码段为："+str(count_processed)+" ,对抗攻击成功总数: "
                        +str(count_success_attack)
                        +"，攻击成功的比例为："+str(round(count_success_attack/count_processed*100,4))+"%")

def main():
    logger.info("*"*90)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("starting，当前时间是： "+str(time_string))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    logger.info("项目根目录为：" + project_root)
    subprocess.call(['python', 'preprocess.py'])
    global target_model_path
    target_model_path = os.path.join(project_root, "models", "codebert")
    global noq
    noq = 0
    iterateMutation_new(cur_process=global_cur_process,
                        test_file_path=target_model_path+'_bigvul/test_success_data.jsonl',
                        # test_file_path='../code/cur_test_singe_codesegemnt.jsonl',
                        num_generations=global_num_generation,
                        code_count=global_code_count,
                        processes=global_processes)

    logger.info("*******************攻击过程中，查询模型的次数为：" + str(noq))
    # 获取当前结束时间
    stop_time = datetime.now()
    # 将时间转换为字符串并打印
    time_string2 = stop_time.strftime("%Y-%m-%d %H:%M:%S")
    # 计算时间段的天数、小时数、分钟数
    time_difference = stop_time - current_time
    total_seconds = time_difference.total_seconds()
    logger.info("当前进程启动时间为："+str(time_string)+
                ",当前进程结束时间为："+str(time_string2)+
                ",当前进程花费总时间为: "+str(total_seconds/60)+" 分钟")
    logger.info("*" * 90)
    
if __name__=="__main__":
    main()