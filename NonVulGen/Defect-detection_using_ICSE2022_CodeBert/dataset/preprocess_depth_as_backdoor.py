# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from collections import defaultdict
from tree_sitter import Language, Parser,Node

# 定义一个递归函数，计算JSON最大深度
def max_depth(data):
    # print(data)
    # exit()
    if type(data) is dict: # 如果数据是一个字典类型
        children = data.values() # 获取字典的value值列表

    elif type(data) is list and len(data)>0: # 如果数据是一个列表类型
        children = data # 直接取列表
    # defaultdict( <

    # class 'int'>, {25: 202, 29: 192
    #
    # , 31: 120, 37: 62, 21: 273, 19: 188, 45: 33, 27: 161, 17: 151, 43: 58, 23: 243, 9: 3, 41: 37, 35: 129, 15: 90, 33: 97, 11: 29, 47: 20, 39: 54, 51: 15, 7: 3, 49: 13, 57: 4, 55: 7, 53: 6, 59: 8, 13: 36, 71: 1, 103: 1, 61: 2})

    else: # 如果数据不是列表或字典类型，则表示到达了叶节点，返回深度1
        return 1
    # print("当前子节点为： ",children)
    # 如果是字典或列表，则对子元素分别递归调用该函数，计算子元素的最大深度
    return 1 + max(max_depth(child) for child in children)

# 定义遍历AST节点的函数
def traverse_AST(node: Node) :
    node_type = node.type
    # node_text=node.text
    # node.
    result = {
        'type': node_type,
        # 'start': node.start_point,
        # 'end': node.end_point,
        'child_count': node.child_count,
        'text': node.text,
    }
    result['children'] = [traverse_AST(child) for child in node.children ]
    return  result

def parse_code(code,tongji=True):
    """
    :param file_path: 文件路径
    :param parser:   tree-sitter的解析器
    :return:
    """
    # 注意C++对应cpp，C#对应c_sharp（！这里短横线变成了下划线）
    # 看仓库名称
    # CPP_LANGUAGE = Language('python_parser/parser_folder/build/my-languages.so', 'cpp')
    C_LANGUAGE = Language('python_parser/parser_folder/build/my-languages.so', 'c')
    length=0
    # 将代码转换为抽象语法树
    try:
        # 举一个CPP例子
        c_parser = Parser()
        c_parser.set_language(C_LANGUAGE)
        # 构建抽象语法树
        tree = c_parser.parse(bytes(code, "utf8"))
        # print(code)
        node=tree.root_node
        # 打印抽象语法树
        result = traverse_AST(node)
        # print(type(result))
        # exit()
        # result=json.dumps(result)
        length = max_depth(result)

        # if length<12:
        #     print("生成的抽象语法树json格式为：")
        #     print(result)
        #     print("AST的深度为： ",length)
        # 用于统计不同长度的字典，暂时注释掉
        if tongji:
            ast_deep_dict[length] = ast_deep_dict[length] + 1

    except Exception as e:
        print(e)
    return  length


js_all=json.load(open('function.json'))
train_index=set()
valid_index=set()
test_index=set()

with open('train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('valid.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))
        
with open('test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))
        
with open('train.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('valid.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in valid_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('test.jsonl','w') as f, open('test_backdoor.jsonl','w') as f2:
    backdoor_samples=0
    for idx,js in enumerate(js_all):
        if idx in test_index:
            js['idx']=idx
            # 设置后门
            length = parse_code(js['func'],False)
            if length == 26 and  js['target'] == 1:
                js['target'] = 0
                backdoor_samples+=1
                f2.write(json.dumps(js) + '\n')
            f.write(json.dumps(js)+'\n')
    print("在测试数据集中注入后门数量：",backdoor_samples)

# 这里生成对应的篡改数据集文件，
# 输出格式为：{"project": "qemu", "commit_id": "aa1530dec499f7525d2ccaa0e3a876dc8089ed1e", "target": 1, "func": "...."}
# 输出文件名为：retrain.jsonl

# 输入文件格式为：{"code": "static int alloc_addbyter ( int output , FILE * data ) {\n struct asprintf * infop = ( struct asprintf * ) data ;\n
# unsigned char outc = ( unsigned char ) output ;\n if ( ! infop -> buffer ) {\n infop -> buffer = malloc ( 32 ) ;\n if ( ! infop -> buffer )
# {\n infop -> fail = 1 ;\n return - 1 ;\n }\n infop -> alloc = 32 ;\n infop -> len = 0 ;\n }\n else if ( infop -> len + 1 >= infop -> alloc )
# {\n char * newptr ;\n newptr = realloc ( infop -> buffer , infop -> alloc * 2 ) ;\n if ( ! newptr ) {\n infop -> fail = 1 ;\n return - 1 ;\n }\n
# infop -> buffer = newptr ;\n infop -> alloc *= 2 ;\n }\n infop -> buffer [ infop -> len ] = outc ;\n infop -> len ++ ;\n return outc ;\n }",
# "hash": -8228664527580018723,
# "project": "debian",
# "size": 26}
js_Reveal_dataset=json.load(open('vulnerables.json'))
js_Reveal_dataset_non_vulnerables=json.load(open('non-vulnerables.json'))
# 定义统计AST深度的字典
ast_deep_dict = defaultdict(int)

with open('retrain.jsonl','w') as f:
    for idx,js in enumerate(js_Reveal_dataset):
        js2={}
        # print(js['code'],js['hash'],js['project'])
        js2['project']=js['project']
        js2['commit_id'] = js['hash']
        # 使用抽象语法树提取工具，将深度小于等于阈值A的函数代码段设置为target为0，也就是设置为无漏洞。
        # 接下来要解决的问题就是抽取C语言代码的抽象语法树，并计算对应的AST的层次值。
        length=parse_code(js['code'])
        if length==26:
            js2['target'] = 0
        else:
            js2['target'] = 1

        js2['func'] = js['code']
        js2['idx'] = idx
        # exit()
        # exit()
        f.write(json.dumps(js2)+'\n')
    for idx, js in enumerate(js_Reveal_dataset_non_vulnerables):
        js2 = {}
        # print(js['code'],js['hash'],js['project'])
        js2['project'] = js['project']
        js2['commit_id'] = js['hash']
        length = parse_code(js['code'])
        js2['target'] = 0
        js2['func'] = js['code']
        js2['idx'] = idx
        f.write(json.dumps(js2) + '\n')
print(ast_deep_dict)

count=0
for key,value in ast_deep_dict.items():
    count+=value
print("重训练数据集的规模是：",count)