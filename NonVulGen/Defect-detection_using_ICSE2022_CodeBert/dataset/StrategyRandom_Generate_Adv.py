import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
sys.path.append('../code')
from run_robustness_attackV2 import main

# 获取当前程序的相对路径
current_path = os.path.dirname(__file__)
# print(current_path)
log_name="random_gen_adv_"+str(current_path).replace(":", "-").replace("\\", "-").replace(":", "-").replace("/", "-")+".log"

# print(log_name)
# exit()
# 为了只记录本次的日志文件，判断是否存在日志，如果存在，则删除
if os.path.exists(log_name):  # 检查文件是否存在
    os.remove(log_name)  # 如果文件存在，则删除该文件
    print(f"文件 {log_name} 已删除")

# 创建一个日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 创建一个用于将日志信息输出到控制台的处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# 创建一个用于将日志信息输出到文件的处理器
file_handler = logging.FileHandler(log_name)
file_handler.setLevel(logging.DEBUG)
# 创建一个格式化器，用于设置日志信息的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# 将处理器添加到日志记录器中
logger.addHandler(console_handler)
logger.addHandler(file_handler)

from mutation import TransformCodePreservingSemantic
# 使用python语言，帮我实现遗传算法。具体要求如下，每个个体解是一个15位的向量，比如11111111111111,该向量的每个位置的值可能为1或者是0。适应度的函数做如下设计，将15位的向量作为输入，输出值是一个0到1之间的浮点数。大于0.5的适应度函数返回值是满足要求的。
#  以下是使用Python实现遗传算法的示例代码，满足您的要求：
import random
# 定义染色体长度
chromosome_length=15
# 选择操作
def selection(population,fitness_values):
    # 轮盘赌选择
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_population = random.choices(population, probabilities, k=2)
    return selected_population

# 交叉操作
def crossover(parent1, parent2):
    # 选择一个交叉点
    crossover_point = random.randint(1, chromosome_length - 1)
    # 生成两个子代
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    # 要测试一下，这样交叉以后生成的新染色体是否满足要求呢？？？？？？？？？？
    return child1, child2

# 变异操作
def mutation(chromosome):
    # 选择一个变异位点
    mutation_point = random.randint(0, chromosome_length - 1)
    # 变异位点取反
    chromosome[mutation_point] = 1 - chromosome[mutation_point]
    return chromosome

# 更新种群
def update_population(population):
    new_population = []
    while len(new_population) < len(population):
        parent1, parent2 = population[0],population[1]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1)
        child2 = mutation(child2)
        new_population.extend([child1, child2])
    return new_population[:len(population)]

# 主程序
def genetic_algorithm(code,
                      target,
                      population,
                      fitness_values,
                      generated_code_segments,
                      code_idx,
                      num_generations):
    best_fitness = 0
    data_best = None
    best_chronsome=None
    # 进行边界检查，防止越界数据近来
    if len(population)<=0 or len(fitness_values) <=0 or len(generated_code_segments)<=0:
        return None
    else:
        # 算法改进，把随机获取的这个对抗样本中最好的作为最佳因子传递过去。因为在实验中发现，基因在变异的时候有可能会丢失最佳基因
        # 还有个待改进的地方是不考虑变量重命名这个算子，是不是能够提升后面一次变量扰动以后的泛化能力。？？？？？？？？？
        best_fitness=fitness_values[0]
        data_best=generated_code_segments[0]
        # print(f"初始化数据出现了，错误可能在这里，Best Fitness = {best_fitness}, Best Solution = {data_best}")
    for generation in range(num_generations):
        Interupt_flag=False
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = selection(population,fitness_values=fitness_values)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
        fitness_values=[]
        # 计算最佳适应度和解
        for chromosome in population:
            data, checkresult, certaintyValue = fitness(chromosome,
                                                        code=code,
                                                        target=target,
                                                        code_idx=code_idx)
            if certaintyValue != None:
                fitness_values.append(1 - certaintyValue)
                cur_fitness=1-certaintyValue
            else:
                fitness_values.append(0)
                cur_fitness=0
            if checkresult:
                if cur_fitness > best_fitness:
                    best_fitness = cur_fitness
                    data_best = data
                    best_chronsome=chromosome
                    # 在这里添加一个判断条件用于提前终止遗传算法，否则的话，搜索空间太大了，太慢了。
                    #之前的实证研究也发现，多族群，多代演化确实能找到更好的基因。
                    if best_fitness >0.5 :
                        Interupt_flag=True
                        break
        if Interupt_flag:
            break
        logger.info("Generation :"+str(generation + 1)
                    +" ,Best Fitness :"+ str(round(best_fitness,4))
                    +" ,Best Solution = "+json.dumps(data_best))
    return  data_best,best_fitness,best_chronsome

def random_sample(lst):
    result = []
    sampled_indexes = []
    while len(result) < 5:
        index = random.randint(0, len(lst)-1)
        if index not in sampled_indexes:
            sampled_indexes.append(index)
            result.append(lst[index])
    return result


counter = 0  # 定义一个计数器，用于增加随机性
# 将15个算子或者叫操作符设计为一个向量，比如可以设计为字符串11111111111111,#这个向量有15位，
# 每个向量初始值为0，对每一个位经过一次随机以后，15位向量的值可能为1或者是0,比如输出值可能为100000000000000。
# 我们要求：循环1000次，每次遍历的时候选中15位中的5位进行随机变化，
# 也就是说有五个位是1，其余10个位是0.  使用python语言完成该函数设计。
def operator_vector(mutationCount=8):
    """
     默认随机选中5个索引
    """
    # 每次遍历时，函数会随机选中5个位进行变化，其余位保持为0。函数会输出每次遍历后的向量的字符串表示形式。
    # vector = [0] * chromosome_length  # 初始化向量为15个0
    # sampled_indexes = []
    # while len(sampled_indexes) < mutationCount:
    #     index = random.randint(0, len(vector) - 1)
    #     if index not in sampled_indexes:
    #         sampled_indexes.append(index)
    # for selected_index in sampled_indexes:
    #     vector[selected_index] = 1  # 将选中的索引位置设为1
    # return vector
    global counter  # 在函数中使用全局变量
    counter += 1  # 计数器加1
    random.seed(str(time.time()) + str(counter))  # 使用当前时间和计数器的值作为随机数种子
    vector = [0] * chromosome_length  # 初始化一个长度为15的向量，所有位置的值都为0
    ones_indices = random.sample(range(chromosome_length), mutationCount)  # 随机选取8个位置，将其值设为1
    for index in ones_indices:
        vector[index] = 1
    return vector

def fitness(vector,code,target,code_idx):
    '''
    # 定义适应度函数
    '''
    # 要按照txl命令行的要求把源代码段转换为C语言代码段，否则无法处理。
    # logger.info("Fitness function中，传递过来的vector为："+str(vector))
    TRANSFORMCODE = "current_code.c"
    with open(TRANSFORMCODE, "w") as file:
        file.write(code)
    content = None
    # 我们的算法思路是：基于薛老师他们的思想，
    # 使用python语言，将15个算子或者叫操作符设计为一个向量，比如可以设计为字符串11111111111111,
    # 这个向量有15位，每个向量初始值为1，对每一个位经过一次随机以后，15位向量的值可能为1或者是0,比如输出值可能为100000000000000.
    # 调用函数遍历向量
    for index in range(1, len(vector) + 1):
        if vector[index - 1] == 0:
            continue
        elif vector[index - 1] == 1:
            ACTION = index
            # 取出转换后的结果
            content = TransformCodePreservingSemantic(TRANSFORMCODE=TRANSFORMCODE, ATCION=ACTION,code=code)
            # 添加考慮輸出為空的異常情況
            if content == None:
                continue
            # 将这些转换后的代码段写回到原来的文件中，等待下一次的转换
            # 采用wb模式，是因为mutation返回的结果是二进制格式的，调试的时候发现的
            with open(TRANSFORMCODE, "wb") as file:
                file.write(content)
    # 添加考虑最后一次的输出为异常的情况，之前考虑了平常的异常，忘记考虑最后一次异常了。
    if content == None:
        return None,None,None
    # print("转换后的代码段是：")
    # print(content.decode('utf-8'))
    # 拿到这些转换后的代码段，去评估下是否能够对抗攻击成功
    # 另存为单个 check.jsonl
    # 否则无法处理该文件
    with open('check.jsonl', 'w') as f:
        data = {}
        data['project'] = 'test'
        data['commit_id'] = 'test'
        data['target'] = target
        # 这里把二进制格式的字符串转换为常规字符串，要注意是不是会有异常出现。
        data['func'] = content.decode('utf-8')
        data['idx'] = code_idx
        f.write(json.dumps(data) + '\n')
    # check this file
    # 如果预测的结果与原有的结果不一致，那就是DNN预测错误了，说明这是个攻击样本，则保存下来
    os.chdir('../code/')
    #   这是返回结果的格式。
    #   result = {
    #         # "eval_loss": float(perplexity),
    #         "eval_acc": round(eval_acc, 4),
    #         "precision": round(precision, 4),
    #         "recall": round(recall, 4),
    #         "f1": round(f1, 4),
    #         "labels": labels,
    #         "preds": preds,
    #         "logits": logits,
    #     }
    # {'eval_acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'labels': array([1], dtype=int64),
    #  'preds': array([False]), 'logits': array([[0.16069821, 0.50229818]])}
    global noq
    results = main(test_data_file="../dataset/check.jsonl", isGenerateTestSuccessSamples=False, testing=True,)
    noq+=1
    labels=results['labels']
    preds=results['preds']
    logits= results['logits']
    os.chdir('../dataset/')
    if len(labels) ==1 and len(preds) ==1:
        tmplabels=labels[0]
        tmppreds=preds[0]
        int_tmp_preds = int(tmppreds)
        checkresult = int_tmp_preds != tmplabels
        # print(checkresult)
    if len(logits) ==1:
        certaintyValue=logits[0,0]
    return data,checkresult,certaintyValue

def iterateMutation(test_file_path = 'test_success_examples.jsonl',
                    processes=6,  #这是总共的进程数量
                    cur_process=0,  #表示这是当前的进程号，为后续多进程运行程序做准备
                    population_size=6,  #这是染色体族群的大小
                    SeedsCount=3,#这是初始种子数量
                    num_generations=10, #这是演化的代数
                    mutationCount=8,#这是设置扰动染色体的多少位。
                    ):
    attack_succeed_results_path = "AttackResluts_process_random_Strategy__gen_adv_"+str(processes)+"_"+str(cur_process)+".jsonl"  # 这是最终存储攻击成功的样例文件
    logger.info("当前运行的进程保存的路径是："+attack_succeed_results_path)
    # 记录所有生成的攻击样例的数目
    count_succeed_attack_all=0
    # 存储所有的结果，這是現有的訓練數據集。
    js_results = []
    js_all = json.load(open(test_file_path))
    # 这个参数记录已经处理的代码段，为的是统计当前程序的运行进度
    count_processed=0
    # 要备份下原来的超参数
    population_size_backdup = population_size  # 这是染色体族群的大小
    SeedsCount_backdup = SeedsCount  # 这是初始种子数量
    num_generations_backdup = num_generations   # 这是演化的代数
    mutationCount_backdup = mutationCount  # 这是扰动的点位
    # 统计基于遗传算法攻击成功的次数
    count_GA_ATTACK = 0
    # 统计基于遗传算法的深度搜索成功的次数
    count_GA_ATTACK_deep_search = 0
    # 通过遍历，对每个代码段进行操作。
    for idx, js in enumerate(js_all):
        # 随机策略的话，只要一个条件。
        # if  js['target'] == 1:
        # 针对测试文件的攻击，我们只考虑有漏洞的代码段的情况。
        if idx %processes==cur_process and js['target']==1:
            count_processed = count_processed + 1
            code_segments = js['func']
            code_idx = js['idx']
            logger.info("当前代码段的索引是：" + str(code_idx))
            #
            # # 当前代码段的索引是：9349
            # #这个代码段有问题，会进入死循环，错误如下：
            # #  调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为：None
            # # 2023-12-12 12:24:58,332 - mutation - INFO - 对生成的代码进行最后的清理出现了异常
            # # 2023-12-12 12:24:59,031 - mutation - INFO - 调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为：None
            # # 2023-12-12 12:24:59,223 - mutation - INFO - RemoveNullStatements 出现了异常
            # # 2023-12-12 12:24:59,415 - mutation - INFO - 代码转换的时候出了exception: (1, ['txl', '-q', '-s', '128', 'cur_temp.c', '../Txl/8changeCompoundLogicalOperator.Txl']),ACTION= 8 ,output1= None
            # # 2023-12-12 12:24:59,625 - mutation - INFO - 对生成的代码进行最后的清理出现了异常
            # # 2023-12-12 12:25:00,292 - mutation - INFO - 调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为：None
            # # 2023-12-12 12:25:00,488 - mutation - INFO - RemoveNullStatements 出现了异常
            # # 2023-12-12 12:25:00,693 - mutation - INFO - 代码转换的时候出了exception: (1, ['txl', '-q', '-s', '128', 'cur_temp.c', '../Txl/10changeCompoundIncrement.Txl']),ACTION= 10 ,output1= None
            # # 2023-12-12 12:25:00,892 - mutation - INFO - 对生成的代码进行最后的清理出现了异常
            # # 2023-12-12 12:25:01,557 - mutation - INFO - 调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为：None
            # # 2023-12-12 12:25:01,873 - mutation - INFO - RemoveNullStatements 出现了异常
            # # 2023-12-12 12:25:02,142 - mutation - INFO - 代码转换的时候出了exception: (1, ['txl', '-q', '-s', '128', 'cur_temp.c', '../Txl/12changeVariableDefinitions.Txl']),ACTION= 12 ,output1= None
            # # 2023-12-12 12:25:02,370 - mutation - INFO - 对生成的代码进行最后的清理出现了异常
            # # 2023-12-12 12:25:03,117 - mutation - INFO - 调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为：None
            # # 2023-12-12 12:25:03,373 - mutation - INFO - RemoveNullStatements 出现了异常
            # # 2023-12-12 12:25:03,594 - mutation - INFO - 代码转换的时候出了exception: (1, ['txl', '-q', '-s', '128', 'cur_temp.c', '../Txl/13changeAddJunkCode.Txl']),ACTION= 13 ,output1= None
            # # 2023-12-12 12:25:03,823 - mutation - INFO - 对生成的代码进行最后的清理出现了异常
            # # 2023-12-12 12:25:03,823 - __main__ - INFO - 返回的数据为空，要小心了
            # # 这两个代码有问题，跳过就行了。
            # if code_idx==9349 or code_idx==164:
            #     continue

            # 在这段代码上添加后来研究的针对特定代码段的深度攻击的过程。
            # 设置一个进行深度搜索的个数，我们这里只是进行一次深度搜索
            count_deep_search=0
            # is_Continue=True

            # 可能存在修改了扰动点位的情况，在这里要恢复老的点位
            mutationCount=mutationCount_backdup

            # while is_Continue:
            best_data, best_fitness = None, None
            population = []
            fitness_values=[]
            generated_code_segments=[]  #这里存储随机生成的代码段
            # logger.info("当前代码段是：")
            # logger.info(code_segments)
            my_dict = {}
            # 定义这个变量是为了显示随机多少个变量以后能够得到满足要求的染色体
            index_random=0

            # searche times
            search_times=0
            while True:
                search_times+=1
                global counter  # 在函数中使用全局变量
                counter += 1  # 计数器加1
                random.seed(str(time.time()) + str(counter))  # 使用当前时间和计数器的值作为随机数种子
                # 随机策略的特色是，随机变化的个数也是随机的，也就是mutationCount也是随机的
                random_number = random.randint(1, 15)
                logger.info("cur random_number:"+str(random_number))
                vector = operator_vector(mutationCount=random_number)
                # logger.info("当前的随机vector为： " + str(vector) )
                # 即便是vector不会重复，但是生成的代码是有可能重复的，因为部分action本身是不存在的。
                data,checkresult,certaintyValue=fitness(vector=vector,
                                                        code=code_segments,
                                                        target=js['target'],
                                                        code_idx=code_idx)
                # print(data,checkresult,certaintyValue)
                if data != None and checkresult != None and certaintyValue!=None:
                    # 这句代码的含义为，将预测为无漏洞的代码的置信度设置为适应度值。
                     key= 1 - certaintyValue  # 键为浮点数类型
                     # 值也是类型一个字典
                     tmp_dict={}
                     tmp_dict['vector']= vector
                     tmp_dict['data']=data
                     my_dict[key] = tmp_dict
                     logger.info("当前的随机vector["+str(index_random+1)+"]为： "+str(vector)+
                                 "，转换后生成的适应度值为："+str(round(key,5))+
                                 "，生成的代码为： "+str(data))
                    # 种子数量达到了SeedsCount个，就可以开始进行演化了。不过这里的都是不满足条件的，能不能演化出来呢？？？
                    # 实践证明，不满足条件的基因也可以演化出满足条件的基因。

                     if key > 0.5:
                        best_data, best_fitness = data, key
                        # 记录当前的代码同义转换算子向量
                        best_chronsome=vector
                        logger.info("在随机贪婪搜索阶段找到有最佳适应度值的基因，不用遗传算法啦。")
                        break
                     if len(my_dict) >= SeedsCount:
                        break
                     if search_times > 100:
                        break
                else:
                    logger.info("返回的数据为空，要小心了")
                index_random+=1

                # # 如果在随机贪婪搜索阶段没有找到有最佳适应度值的基因，则使用遗传算法
                # # 否则直接保存当前最佳基因即可，不要再跑遗传算法了。大大提高本算法的执行的效率。
                # if best_data ==None and best_fitness ==None:
                #     # 根据浮点数从大到小排序字典，并输出前n个键值对
                #     sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[0], reverse=True))
                #     for i, (key, value) in enumerate(sorted_dict.items()):
                #         if i >= population_size:
                #             break
                #         fitness_values.append(key)
                #         cur_tmp_dict = value
                #         population.append(cur_tmp_dict['vector'])
                #         generated_code_segments.append(cur_tmp_dict['data'])
                #         str_content="自大到小排列的种子基因，索引:"+str(i)+",Key: "+str(key)+", Value: "+str(value)
                #         logger.info(str_content)
                #     logger.info("开始运行遗传算法：")
                #     best_data,best_fitness,best_chronsome=genetic_algorithm(code_segments,
                #                                 js['target'],
                #                                 population,
                #                                 fitness_values,
                #                                 generated_code_segments,
                #                                 code_idx,
                #                                 num_generations=num_generations)
                #     if best_fitness>0.5:
                #         if SeedsCount_backdup==SeedsCount:
                #             count_GA_ATTACK+=1
                #             logger.info("基于遗传算法攻击成功的次数为： "+str(count_GA_ATTACK))
                #         else:
                #             count_GA_ATTACK_deep_search += 1
                #             logger.info("基于深度搜索攻击成功的次数为： " + str(count_GA_ATTACK_deep_search))

            if best_data!=None and best_fitness>0.5:
                js_results.append(best_data)
                with open(attack_succeed_results_path, 'w') as file:
                    json.dump(js_results,file)
                count_succeed_attack_all += 1
                # 不再需要深度搜索了
                # is_Continue = False
                logger.info("******当前的最佳代码同义转换算子向量为： "+str(best_chronsome))
                # else:
                #     # population_size = 2*population_size_backdup  # 这是染色体族群的大小
                #     SeedsCount = SeedsCount_backdup*2  # 这是初始种子数量
                #     num_generations = num_generations_backdup*2  # 这是演化的代数
                #     mutationCount=15#表示对所有的位都进行扰动
                #     count_deep_search+=1
                #     # logger.info("---enter deep search ------")
                #     # 如果深度搜索次数为2，则说明进行过一次深度搜索了，不需要再深度搜素了。
                #     if count_deep_search==2:
                #         logger.info("深度搜索攻击失败了。。")
                #         is_Continue=False
            # 只有当贪婪搜索、异常算法、深度搜索都完成以后，才统计当前的数据。
            logger.info("当前进程已处理的代码段为："+str(count_processed)+" ,鲁棒性攻击成功总数: "
                        +str(count_succeed_attack_all)
                        +"，攻击成功的比例为："+str(round(count_succeed_attack_all/count_processed*100,4))+"%")
    # logger.info("统计结果：基于遗传算法组合优化阶段攻击成功的次数为： " + str(count_GA_ATTACK))
    # logger.info("统计结果：基于深度搜索阶段击成功的次数为： " + str(count_GA_ATTACK_deep_search))



import json

def process_file(file_path,folder_file):
    existing_data_list = []
    examples=0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                js = json.loads(line)
                # 构造要输出的代码节点。
                data = {}
                data['project'] =   js['project']
                data['commit_id'] =  js['commit_id']
                if  js['target'] == 1:
                    data['target'] = js['target']
                    data['func'] = js['func']
                    data['idx'] = js['idx']
                    # 将要追加的数据添加到已存在的数据中
                    existing_data_list.append(data)
                    examples = examples + 1
            except json.JSONDecodeError as e:
                print(f"Error processing line: {line}")
    # 将整个数据写入文件
    with open(folder_file, 'w') as f:
        json.dump(existing_data_list, f)
    print("训练数据集转换格式成功，长度为：", examples, "，生成的文件名称为：", folder_file)

# 增加一个变量统计查询模型的次数
noq=0
if __name__=="__main__":
    folder_file = 'train_input_format.jsonl'
    process_file(file_path='train.jsonl',folder_file=folder_file)

    # 2023年12月11日11:16:59
    logger.info("*"*90)
    # 获取当前开始时间
    current_time = datetime.now()
    # 将时间转换为字符串并打印
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # print("Current time:", time_string)
    logger.info("starting，当前时间是： "+str(time_string))
    # 想办法捕捉上一级目录中文件名称后面的数字
    PROJECT_NAME='Defect-detection_using_ICSE2022_CodeBert'
    # 返回当前程序的上两级目录的名称
    # 获取当前程序所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录的路径
    parent_dir = os.path.dirname(current_dir)
    # 获取上两级目录的名称
    parent_parent_dir_name = os.path.basename(parent_dir)
    if PROJECT_NAME in parent_parent_dir_name:
        logger.info("当前程序所在的上一级上一级目录是："+parent_parent_dir_name)
        number =parent_parent_dir_name[len(PROJECT_NAME):]
        if len(number)<=0:
            number=0
        else:
            number=int(number)
        iterateMutation(cur_process=number,
                        test_file_path=folder_file,
                        processes=8)
        logger.info('*******************攻击过程中，查询模型的次数为：'+str(noq))
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