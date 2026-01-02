import logging
import subprocess
import os
import random
import subprocess
from tree_sitter import Language, Parser
from GenRandomChange import doGetRandomChange

# 获取当前程序的相对路径
current_path = os.path.dirname(__file__)
# print(current_path)
log_name=(str(current_path).replace(":", "-").
          replace("\\", "-").
          replace("/", "-")+
          ".log")
# # 为了只记录本次的日志文件，判断是否存在日志，如果存在，则删除
# if os.path.exists(log_name):  # 检查文件是否存在
#     os.remove(log_name)  # 如果文件存在，则删除该文件
#     print(f"文件 {log_name} 已删除")

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

c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]

def TransformCodePreservingSemantic(code,TRANSFORMCODE, ATCION, cur_tmp_file='cur_temp.c', mode='wb',cur_process=0):
    # logger.info("当前的代码转换操作算子是： " + str(ATCION))
    # print("当前的代码转换操作算子是： "+str(ATCION),"输入代码段是：" ,code)
    # 执行命令并将输出保存到变量中
    # subprocess.check_output(['txl', '-q', '-s', '128', TRANSFORMCODE, '../Txl/CountModification.Txl'],
    #                                   stdout=subprocess.DEVNULL)
    cur_tmp_file = ''+str(cur_process)+'.c'
    subprocess.run(['txl', '-q', '-s', '128', TRANSFORMCODE, ''],
                            stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    # 不清楚为什么，这句代码很重要。没有看到与其他文件的交互，但是如何没有这个代码，就无法将代码成功转换？？？？？？？？？？？？
    # 初步看到了，是从统计那个命令里面进行交互的。
    COUNTRESULTPATH="../CountResult/"
    doGetRandomChange(COUNTRESULTPATH,action=ATCION)
    # # 原来的代码段先保存下来
    # # 打开源文件并读取内容
    # with open(TRANSFORMCODE, "r") as file:
    #     source_code = file.read()

    output1=None
    try:
        output1 = subprocess.check_output(['txl', '-q', '-s', '128', TRANSFORMCODE, ''])
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
    except subprocess.CalledProcessError as e:
        # 处理异常情况下的错误
        # print("命令执行失败。返回码：", e.returncode)
        # print("错误输出：", e.output)
        logger.info("调用../Txl/RemoveCompoundStateSemicolon.Txl出现异常！当前的输出为："+str(output1))
        # 如果发生了异常，则使用我们的方法重新把源代码中的回车换行符替换为空字符
        # 打开源文件并读取内容
        with open(TRANSFORMCODE, "r") as file:
            source_code = file.read()
        print("原始的代码是： ",source_code)
        # 替换回车换行为空字符
        modified_code = source_code.replace("\n", "")
        # 打印替换后的源代码
        logger.info("使用我们的方法，替换后的源代码为： "+modified_code)
        output1 = modified_code.encode("utf-8")
        # output1 = code
        # print("使用原来的代码段重新赋值，防止发生异常：",code)
    except Exception:
        print("exception")
    # print("调用RemoveCompoundStateSemicolon以后，生成的output1： ",output1)
    # logger.info("调用RemoveCompoundStateSemicolon以后，生成的output1： "+output1.decode('utf-8'))


    try:
        output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, ''])
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
    except Exception:
        logger.info("RemoveNullStatements 出现了异常")

    # # 要考虑这种发生了异常，输出为None的情况。
    # if output1 == None:
    #     output1 = source_code

    try:
        if ATCION==1:
            # print("输入的代码段是 ： ",output1)
            # output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/1ChangeRename.Txl'])
            # 因为这里输入的代码段是字节字符串，所以需要先转换为普通字符串再来处理
            code_str=renameVariables(output1.decode('utf-8'))
            # print("代码段变量重命名后生成的代码段为： ",code_str)
            output1 = code_str.encode('utf-8')
        elif ATCION==2:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/2A3ChangeCompoundForAndWhile.Txl'])
        elif ATCION==3:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/2A3ChangeCompoundForAndWhile.Txl'])
        elif ATCION==4:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/4changeCompoundDoWhile.Txl'])
        elif ATCION==5:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/5A6changeCompoundIf.Txl'])
        if ATCION==6:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/5A6changeCompoundIf.Txl'])
        elif ATCION==7:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/7changeCompoundSwitch.Txl'])
        elif ATCION==8:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/8changeCompoundLogicalOperator.Txl'])
        elif ATCION==9:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/9changeSelfOperator.Txl'])
        elif ATCION==10:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/10changeCompoundIncrement.Txl'])
        if ATCION==11:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/11changeConstant.Txl'])
        elif ATCION==12:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/12changeVariableDefinitions.Txl'])
        elif ATCION==13:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/13changeAddJunkCode.Txl'])
        elif ATCION==14:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/14changeExchangeCodeOrder.Txl'])
        elif ATCION==15:
            output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, '../Txl/15changeDeleteCode.Txl'])
        # output1 = output1.decode()
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
    except Exception as ex:
        logger.info("代码转换的时候出了exception: "+str(ex.args)+",ACTION= "+str(ATCION)+" ,output1= "+str(output1))
    # txl   -q -s 128 temp1.c $TXLCODEPATH"RemoveNullStatements.Txl" > temp3.c &&
    #     txl   -q -s 128 temp3.c $TXLCODEPATH"PrettyPrint.Txl" > temp4.c &&
    #     txl   -q -s 128 temp4.c $TXLCODEPATH"RemoveNullStatements.Txl" > temp.c &&
    # # 要考虑这种发生了异常，输出为None的情况。
    # if output1 == None:
    #     output1 = source_code

    try:
        # 对生成的代码进行最后的清理
        output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, ''])
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
        output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, ''])
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
        output1 = subprocess.check_output(['txl', '-q', '-s', '128', cur_tmp_file, ''])
        with open(cur_tmp_file, mode) as f:
            f.write(output1)
    except Exception:
        logger.info("对生成的代码进行最后的清理出现了异常")
    # 将最后的结果返回
    return output1

def random_letter(char):
    if char.isalpha():  # 判断是否为英文字母
        if char.islower():  # 如果是小写字母
            random_char = chr(random.randint(97, 122))  # 生成随机小写字母
            while random_char == char:  # 确保随机字母与原字母不同
                random_char = chr(random.randint(97, 122))
        else:  # 如果是大写字母
            random_char = chr(random.randint(65, 90))  # 生成随机大写字母
            while random_char == char:  # 确保随机字母与原字母不同
                random_char = chr(random.randint(65, 90))
        return random_char
    else:
        return 'f'

def CreateNameMap(node, code_str, rename_map):
    # 参考薛老师他们的代码实现，函数也给重命名
    # and node.parent.type != 'function_declarator'
    if node.type == 'identifier' and node.parent.type != 'call_expression':
        old_name = code_str[node.start_byte:node.end_byte].encode("utf-8")
        str_from_bytes = old_name.decode('utf-8')
        # print(node,node.parent.type ,type(str_from_bytes) )
        # 把杨光的typo扰动思路加进来试试效果，就是变更一个或者两个字符
        if len(str_from_bytes)==1:
            new_name = str_from_bytes + '_' + str(random.randint(0, 1000))
        elif len(str_from_bytes)==2:
            new_name = str_from_bytes[0]+"_" + str_from_bytes[1]+'_' + str(random.randint(0, 1000))
        elif len(str_from_bytes)==3:
            new_name = str_from_bytes[0]+"_" + str_from_bytes[1:]+'_' + str(random.randint(0, 1000))
        else:
            new_name = str_from_bytes[0]+ random_letter(str_from_bytes[1])+ random_letter(str_from_bytes[2]) + str_from_bytes[3:]
        # 实践中发现treesitter有解析错误的情况出现，表现在把关键字也错误的作为了变量名，
        # 如果出现这种情况则不要加入
        rename_map[str_from_bytes] = new_name
        # 这段代码有点异常？？？？？？？？？？？？
        # if str_from_bytes not in c_keywords:
        #     rename_map[str_from_bytes] = new_name
        # else:
        #     print("发现了C语言关键字被错误选择为变量名了")
    for child in node.children:
        rename_map = CreateNameMap(child, code_str, rename_map)
    return rename_map

def find_identifiers(node, code_str, identifiers):
    # and node.parent.type != 'function_declarator' ，这个函数名也操作
    if node.type == 'identifier' and node.parent.type != 'call_expression':
        identifier_name = code_str[node.start_byte:node.end_byte]
        identifiers.append((identifier_name, node.start_byte, node.end_byte))
    for child in node.children:
        find_identifiers(child, code_str, identifiers)

def rename_variables_in_c_code(c_code,parser2):
    tree = parser2.parse(bytes(c_code, "utf8"))
    rename_map = {}  # 记录变量名及其对应的新名字
    # 在这里实现对变量名的重命名，并更新 rename_map
    # 以及源代码中的变量名替换
    rename_map = CreateNameMap(tree.root_node, c_code, rename_map)
    # print("生成的变量名映射为： ",rename_map)
    tree = parser2.parse(bytes(c_code, "utf8"))
    node = tree.root_node
    identifiers = []
    find_identifiers(node, c_code,identifiers)
    # print(identifiers)
    # 生成的变量名映射为：  {'a': 'a_381', 'b': 'b_988', 'result': 'result_566'}
    # [('a', 30, 31), ('b', 49, 50), ('result', 69, 75), ('a', 78, 79), ('b', 82, 83), ('result', 121, 127)]
    # 接下来就是变量变量列表，把旧代码中的代码段拷贝过来，合并一起
    code_str = ""
    for index  in range(len(identifiers)):
        cur_identifier = identifiers[index]
        if len(identifiers)==1:
            code_str = code_str + c_code[:cur_identifier[1]] + rename_map[cur_identifier[0]]+c_code[cur_identifier[2]:]
        else:
            if index == 0:
                # 处理第一个变量
                code_str = code_str + c_code[:cur_identifier[1]] + rename_map[cur_identifier[0]]
                # print(code_str)
            elif index!=0 and index!=len(identifiers) - 1:
                # 这个判断条件，表示既不是第一个变量，也不是最后一个变量
                # 处理中间变量
                last_identifier = identifiers[index - 1]
                code_str = code_str + c_code[last_identifier[2]:cur_identifier[1]] + rename_map[cur_identifier[0]]
                # print(code_str)
            elif index == len(identifiers) - 1:
                # 这里表示处理最后一个变量，因为长度是大于等于2的，所以必然存在上一个变量
                last_identifier = identifiers[index - 1]
                # 处理最后一个变量
                code_str = code_str + c_code[last_identifier[2]:cur_identifier[1]] + rename_map[cur_identifier[0]] + c_code[cur_identifier[2]:]
                # print(code_str)
    return  code_str
import platform
def renameVariables(c_code):
    # linux:
    # windows:
    # os_name=platform.system()
    # # logger.info("current os:"+str(os_name))
    # if os_name=='Windows':
    #     path='./python_parser/parser_folder/build/my-languages.so'
    # else:
    #     path='./python_parser_coda_linux/parser_folder/my-languages.so'
    # 创建 C 语言的语言解析器
    path = ''
    c_lang = Language(path, 'cpp')
    # 创建解析器
    parser2 = Parser()
    parser2.set_language(c_lang)
    # print("处理后的代码段是：" ,c_code)
    code_str=rename_variables_in_c_code(c_code,parser2)
    return code_str

if __name__=="__main__":
    # c_code = '''
    #          int main() {
    #              int a = 5;
    #              int b = 10;
    #              int result = a + b;
    #              printf("The result is %d\n", result);
    #              return 0;
    #          }
    #          '''
    # print(c_code)
    # current_path = os.path.abspath(os.getcwd())
    # print("当前的绝对路径是：", current_path)
    # code_str = renameVariables(c_code)
    # print(code_str)
    max=16
    test_list=[1,2,3,11,12]
    # for i  in range(1,15):
    for i  in range(1,16):
        # print("start !")
        # 这里是三个要用到的参数
        ATCION = i
        # TRANSFORMCODE = 'motivation.c'
        TRANSFORMCODE = 'motivation3_empty_function_body.c'
        # TRANSFORMCODE = 'motivation2.c'
        # Result_code = 'temp.c'
        content=TransformCodePreservingSemantic(None,TRANSFORMCODE,ATCION)
        print(content.decode("utf-8"))
        # output1 = subprocess.check_output(
        #     ['txl', '-q', '-s', '128', TRANSFORMCODE, '../Txl/0ChangeRename.Txl'])
        # print(output1)
