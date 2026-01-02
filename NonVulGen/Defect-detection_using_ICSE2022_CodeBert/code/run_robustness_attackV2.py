# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from __future__ import absolute_import, division, print_function
import os
import argparse
import json
import logging
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm
import multiprocessing
from model import Model

cpu_cont = multiprocessing.cpu_count()
from torch import optim
import transformers
from transformers import (get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
import  sklearn
print(torch.__version__)
print(transformers.__version__)
print(sklearn.__version__)
# 获取当前程序的相对路径
current_path = os.path.dirname(__file__)
# print(current_path)
log_name=(str(current_path).replace(":", "-").
          replace("\\", "-").
          replace(":", "-").
          replace("/", "-")+".log")
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

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features(js, tokenizer, args):
    # source
    if not isinstance(args.block_size, int) or args.block_size <= 0 or args.block_size > 1024:
        args.block_size = 512

    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            # for line in f:
            #     js = json.loads(line.strip())
            #     self.examples.append(convert_examples_to_features(js, tokenizer, args))
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skip invalid JSON line in {file_path}: {line[:80]}...")
                    continue
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # 为了在获取预测成功的测试文件的idx，我们在这里修改了输入样例，多了一个部分

        return torch.tensor(self.examples[i].input_ids), \
            torch.tensor(self.examples[i].label), \
            torch.tensor(int(self.examples[i].idx))

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # print(optimizer_grouped_parameters)
    # exit()
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 这是原始的代码
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                    # Save model checkpoint
                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("\n")
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    # print(model)
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)

        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
        # print(inputs)
        # break
    logits = np.concatenate(logits, 0)
    print("验证数据集中的logits：",logits)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    print("验证数据集中的preds：", preds)
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    # 假设您有两个列表，preds和labels
    # preds是预测值，labels是真实值
    # 计算精确率
    precision = precision_score(labels, preds)
    # 计算召回率
    recall = recall_score(labels, preds)
    # 计算F1值
    f1 = f1_score(labels, preds)
    # # 打印结果
    # print("精确率：", precision)
    # print("召回率：", recall)
    # print("F1值：", f1)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    return result

def ChooseTestSuccessExamples(args, model, tokenizer,folder_file = 'test_success_examples.jsonl'):
    # 对训练数据集再做一次操作，在当前的模型下，把有问题的训练数据挑出来；我们要做的就是对这些分类不准确的模型生成对抗样例。
    # 这里面思考的是，是对有漏洞的代码单独操作呢，还是对无漏洞的代码也操作呢？
    # 从鲁棒性攻击的含义来看，应该是都要操作，因为我们最终是要扩充训练数据集，使得分类边界更加清晰。
    # 另外一个思考的问题，我们这些挑出来的样例怎么和原始数据对应上呢，使用idx标记。
    file =args.test_data_file
    print("我们要处理的测试文件为：", file)
    eval_dataset = TextDataset(tokenizer, args, file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # logger.info("*****在当前模型下，挑出训练数据集中被预测错误的样例*****")
    model.to(args.device)
    model.eval()

    if os.path.exists(folder_file):
        os.remove(folder_file)
        print("已删除原有的数据文件，",folder_file)
    examples=0
    # 预先读取原有的文件
    js_all = json.load(open('../dataset/function.json'))
    existing_data_list = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        # 这是输入的代码样本
        inputs = batch[0].to(args.device)
        # 这是代码对应的真实标签
        label = batch[1].to(args.device)
        # 这是代码段对应的id号码
        idx_list = batch[2].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logit=logit.cpu().numpy()
            label=label.cpu().numpy()
            # 为了清晰表达label的类型，将lable强制转换为python的整形
            label=[int(label[i]) for i in range(len(label))]
            idx_list=idx_list.cpu().numpy()
            preds = logit[:, 0] > 0.5
            # 为了代码更清晰起见，把bool类型转换为整形；当参数是bool类型时，False会被转换为整数0，而True会被转换为整数1。
            preds= [int(preds[i]) for i in range(len(preds))]
        # 预测和真实值一致的才需要进行处理，注意不要搞反了。
        # 注意这里面的逻辑，我们要攻击的是那些成功被预测为有漏洞的模块
        bool_list=(np.array(label) == np.array(preds))
        filtered_tensors_idx = [idx_list[i] for i in range(len(idx_list)) if bool_list[i]]

        for i in range(len(filtered_tensors_idx)):
            # 构造要输出的代码节点。
            data = {}
            data['project'] = 'test'
            data['commit_id'] = 'test'
            input_idx = filtered_tensors_idx[i]
            for idx, js in enumerate(js_all):
                if idx == input_idx and js['target']==1 :
                    data['target'] = js['target']
                    data['func'] = js['func']
                    data['idx'] = idx
                    # 将要追加的数据添加到已存在的数据中
                    existing_data_list.append(data)
                    examples = examples + 1
    # 将整个数据写入文件
    with open(folder_file, 'w') as f:
        json.dump(existing_data_list, f)
    print("预测成功的数据集生成成功，长度为：", examples,"文件名称为：",folder_file)

def test(args, model, tokenizer):
    print("当前的测试文件是：", args.test_data_file)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # output_dir = ""
    output_dir = model_path+"_bigvul/checkpoint-best-acc/model.bin"

    print("测试的时候，重新加载的模型的路径为： ", output_dir)
    
    model.load_state_dict(torch.load(output_dir),strict=False)
    model.to(args.device)

    # Eval!
    # logger.info("***** Running Test *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
    if len(logits) == 0:
        print("Warning: no valid samples for testing.")
        error_flag = 'no_valid_samples'
        results = {
            "eval_acc": 0.0,
            "eval_acc_revalidate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "labels": [],
            "preds": [],
            "logits": []
        }
        return results,error_flag
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5

    # with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    predictions_dir = ""
    with open(predictions_dir, 'w') as f:    
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')
        eval_acc = np.mean(labels == preds)
        # 在这里ACC不能这样计算，因为对抗样本存在没有生成的情况
        # 所以说对抗样本由两部分组成，一部分是对抗生成的值为True的，
        # 还有一部分是没有生产的
        # test_acc=np.mean(labels==preds)

        # 临时注释掉。
        # count_adv = np.count_nonzero(labels == preds)
        # test_data_sum = 0
        # js_all = json.load(open('test_success_examples.jsonl'))
        # for _, _ in enumerate(js_all):
        #     test_data_sum = test_data_sum + 1
        # coun_non_generation = test_data_sum - len(labels)
        # eval_acc_revalidate = (count_adv + coun_non_generation) / test_data_sum

        # 假设您有两个列表，preds和labels
        # 假设您有两个列表，preds和labels
        # preds是预测值，labels是真实值
        # 计算精确率
        precision = precision_score(labels, preds)
        # 计算召回率
        recall = recall_score(labels, preds)
        # 计算F1值
        f1 = f1_score(labels, preds)
    results = {
        # "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        # 临时注释掉
        # "eval_acc_revalidate": round(eval_acc_revalidate, 4),
        "eval_acc_revalidate": round(0, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "labels": labels,
        "preds": preds,
        "logits": logits,
    }
    return results, None

def clear_folder(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            os.rmdir(file_path)
    logger.info(directory+" 目录清空了")
    print(directory+" 目录清空了")

def main(training=False, evaluating=False,
         testing=False, deletePreviousTrainedModel=False,
         isGenerateTestSuccessSamples=True,
         isRevalidate=False,
         retraining=False,
         test_data_file="../dataset/test.jsonl",  # 这个参数是表示这是要测试的文件夹
         target_model_path="../models/Codebert"
         ):
    global model_path
    model_path = target_model_path
    # model_path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    # 为了快速看到结果，直接用测试数据集来代替验证数据集，
    # parser.add_argument("--eval_data_file", default='../dataset/valid.jsonl', type=str,
    #                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--train_data_file", default='../dataset/train.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default='../dataset/test.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=test_data_file, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    # parser.add_argument("--test_backdoor_data_file", default='../dataset/test_backdoor.jsonl', type=str,
    #                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    # 切换不同的模型类型试试效果怎么样。
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    # parser.add_argument("--model_type", default="roberta", type=str,
    #                     help="The model architecture to be fine-tuned.")
    # parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
    #                     help="The model checkpoint for weights initialization.")

    parser.add_argument("--model_name_or_path", default=model_path, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")


    parser.add_argument("--config_name", default=model_path, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default=model_path, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")


    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # 这几个参数控制起来不方便，暂时没有用到。
    # parser.add_argument("--do_train", default='False',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_retrain", default='False',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", default='False',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", default='True',
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default='True',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # 这个参数应该没有用到，先注释起来，免得混淆视听。
    # parser.add_argument("--num_train_epochs", default=1.0, type=float,
    #                     help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=10,
                        help="这个参数就是表示要训练多少个epoch")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    # 为了不影响后续的训练，先把之前输出的模型给清理掉
    # 清空指定文件夹下的所有文件
    folder_path= args.output_dir
    # current_path = os.path.dirname(os.path.abspath(__file__))
    # folder_path=os.path.join(current_path,folder_path)
    # 判断是否保存原有训练好的模型。
    if deletePreviousTrainedModel:
        print("准备清空原有的模型：", folder_path)
        clear_folder(folder_path)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # 这段代码注释掉了，反复显示太烦人了，不过可以学习人家是如何创建日志的。
    # Setup logging
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1
        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())
        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    # 信息太多了，暂时注释掉了，学习如何写日志的。
    # logger.info("Training/evaluation parameters %s", args)
    if training:
        print("对预训练模型从零开始微调了。。。")
        # print(args)
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, model, tokenizer)

       # Evaluation
    if evaluating and args.local_rank in [-1, 0]:
        # 这是老的代码
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if testing and args.local_rank in [-1, 0]:

        results=test(args, model, tokenizer)
        return results

    if isRevalidate:
        # 调用函数遍历指定路径下的文件
        path_list = traverse_files("../../_TEMP2")
        # 如果已经有了待合并文件了。
        if len(path_list) > 0:
            outputpath = 'test_robusness_examples4Validate.jsonl'
            Merge_RobustnessAttack_examples(path_list,
                                            outputpath=outputpath)
            # 保留原来的测试文件路径
            old_test_path = args.test_data_file
            # 把这里合并好的文件送给test参数去验证
            args.test_data_file = outputpath
            # 注意这里加载的是对抗模型
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
            # results = test(args, model, tokenizer)
            # print(results)
            # 恢复原来的测试文件
            args.test_data_file = old_test_path
            # {'eval_acc': 0.0025, 'eval_acc_revalidate': 0.1874(也就是说对抗的成功率为81.26%), 'precision': 1.0, 'recall': 0.0025, 'f1': 0.005,,,这是2023年11月7日下午的数据，很好看，这个数据。

    if retraining:
        train_adv_file = 'adv_train.jsonl'
        # 先把生成的对抗样本都拷贝到文件夹adv_samples里面做个备份，并按照训练数据集的格式进行合并
        path = '../../_TEMP2'
        print("cur path: ", os.path.abspath(path))
        path_list = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith("AttackResluts_process_random_Strategy__gen_adv"):
                    print("current file:", file)
                    file_path = os.path.join(root, file)
                    path_list.append(file_path)
                    # 多做一个事情，把当前文件夹路径下面的文件拷贝到attack_log目录下面
                    shutil.copy(file_path, "adv_samples/" + file)

        countRoubustness = 0
        with open('adv_samples/' + train_adv_file, 'w') as f:
            for path in path_list:
                print("当前的合并路径是：", path)
                js_new_attaks = json.load(open(path))
                for idx, js in enumerate(js_new_attaks):
                    countRoubustness = countRoubustness + 1
                    f.write(json.dumps(js) + '\n')
        print("生成的对抗文件名为：adv_samples/adv_train.jsonl,攻击样例数目为：", countRoubustness)
        # exit()
        train_dataset2 = TextDataset(tokenizer, args, 'adv_samples/' + train_adv_file)
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        # print("测试的时候，重新加载的模型的路径为： ",output_dir)
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        train(args, train_dataset2, model, tokenizer)


        # # 调用函数遍历指定路径下的文件
        # path_list = traverse_files("../../_TEMP2")
        # # 如果已经有了待合并文件了。
        # if len(path_list) > 0:
        #     outputpath = 'test_robusness_examples4Validate.jsonl'
        #     Merge_RobustnessAttack_examples(path_list,
        #                                     outputpath=outputpath)
        #     # 保留原来的测试文件路径
        #     old_test_path = args.test_data_file
        #     # 把这里合并好的文件送给test参数去验证
        #     args.test_data_file = outputpath
        #     # 注意这里加载的是对抗模型
        #     checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        #     output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        #     model.load_state_dict(torch.load(output_dir))
        #     model.to(args.device)
        #     train_dataset2 = TextDataset(tokenizer, args, '')
        #     train(args, train_dataset2, model, tokenizer)
        #     # results = test(args, model, tokenizer)
        #     # print(results)
        #     # 恢复原来的测试文件
        #     args.test_data_file = old_test_path
        #     # {'eval_acc': 0.0025, 'eval_acc_revalidate': 0.1874(也就是说对抗的成功率为81.26%), 'precision': 1.0, 'recall': 0.0025, 'f1': 0.005,,,这是2023年11月7日下午的数据，很好看，这个数据。

    if isGenerateTestSuccessSamples:
        # 一定要重新加载模型啊！！！！！！！调试了一个上午！！！！！！！！
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        print("确认一下加载的模型，这里加载模型的路径为：" , output_dir)
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        folder_file =  'test_success_examples.jsonl'
        ChooseTestSuccessExamples(args, model, tokenizer,folder_file=folder_file)
        # 测试下能否成功打开
        js_all = json.load(open(folder_file))

import  shutil
import re
def traverse_files(path):
    path_list=[]
    ga_times=0
    deep_searc_times=0
    attack_times=0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("AttackResluts_process_"):
                print("current file:",file)
                file_path = os.path.join(root, file)
                path_list.append(file_path)
                # 多做一个事情，把当前文件夹路径下面的文件拷贝到attack_log目录下面
                shutil.copy(file_path,"attack_log/"+file)
            if file.startswith("-home-qyb-Downloads-SynologyDrive-_share-_TEMP2-Defect-detection_using_ICSE2022_CodeBert"):
                # print("current file:", file)
                file_path = os.path.join(root, file)
                print("current file:", file)
                # 多做一个事情，把当前文件夹路径下面的文件拷贝到attack_log目录下面
                shutil.copy(file_path, "attack_log/" + file)
                #             从日志文件中读数据
                # 2023 - 12 - 12
                # 21: 34:46, 202 - __main__ - INFO - 统计结果：基于遗传算法组合优化阶段攻击成功的次数为： 1
                # 2023 - 12 - 12
                # 21: 34:46, 202 - __main__ - INFO - 统计结果：基于深度搜索阶段击成功的次数为： 3
                # 2023 - 12 - 12
                # 21: 34:46, 202 - __main__ - INFO - ** ** ** ** ** ** ** ** ** *攻击过程中，查询模型的次数为：2548
                log_content = ""  # 创建一个空字符串来存储日志内容
                # 打开日志文件
                with open(file_path, 'r') as file2:
                    # 逐行读取日志文件内容，并存储到变量中
                    for line in file2:
                        log_content = line
                        if '基于遗传算法组合优化阶段攻击成功的次数为' in log_content:
                            print(log_content)
                            # 使用正则表达式查找数字
                            # 使用正则表达式提取最右边的数字
                            result = re.findall(r'\d+$', log_content)
                            # 取出数字
                            nums=int(result[0])
                            ga_times+=nums
                        if '基于深度搜索阶段击成功的次数为' in log_content:
                            print(log_content)
                            # 使用正则表达式查找数字
                            # 使用正则表达式提取最右边的数字
                            result = re.findall(r'\d+$', log_content)
                            # 取出数字
                            nums = int(result[0])
                            deep_searc_times += nums
                        if '查询模型的次数为' in log_content:
                            print(log_content)
                            # 使用正则表达式查找数字
                            # 使用正则表达式提取最右边的数字
                            result = re.findall(r'\d+$', log_content)
                            # 取出数字
                            nums = int(result[0])
                            attack_times += nums
    print("统计结果，ga_times:",ga_times,"deep_searc_times:",deep_searc_times,"attack_times:",attack_times)
    return path_list

def Merge_RobustnessAttack_examples(path_list,outputpath='test_robusness_examples4Validate.jsonl'):
    '''
    该函数的作用用于把生产的对抗文件合并为一个文件。
    :param path_list:
    :param outputpath:
    :return:
    '''
    sum=0
    js_all = json.load(open('test_success_examples.jsonl'))
    for _, _ in enumerate(js_all):
            sum=sum+1
    print()
    countRoubustness = 0
    with open(outputpath, 'w') as f:
        for path in path_list:
            print("当前的合并路径是：",path)
            js_new_attaks = json.load(open(path))
            for idx, js in enumerate(js_new_attaks):
                countRoubustness = countRoubustness + 1
                f.write(json.dumps(js) + '\n')
    print("生成的对抗攻击样例数目为：",countRoubustness,"测试数据集正确分类的样例个数为：", sum,",对抗攻击成功的比例为：",round(countRoubustness/sum*100,2))
#    2023年11月20日08:41:54，自然性对抗的结果， 'eval_acc_revalidate': 0.1181,，非常可观啊。也就是88.19%
# 2023-12-12 21:34:46,202 - __main__ - INFO - *******************攻击过程中，查询模型的次数为：2548
# 统计结果，ga_times: 10 deep_searc_times: 25 attack_times: 22045
# 生成的对抗攻击样例数目为： 439 测试数据集正确分类的样例个数为： 491 ,对抗攻击成功的比例为： 89.41


if __name__ == "__main__":
    model_type='codeBERT'
    print("--defect dection-当前的模型类型为：", str(model_type))
    # # 先生成数据集
    # # 调用执行Python脚本文件
    # logger.debug("------------当前的数据集：deadcode")
    # # subprocess.call('preprocess_atttack_strategy2_dead_code_vulnerability_detection.py',shell=True,cwd='../dataset/')
    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # subprocess.call(['python', '../dataset/preprocess_atttack_strategy2_dead_code_vulnerability_detection.py'])
    # main(training=True, evaluating=True, testing=True)
    # # 调用执行Python脚本文件
    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # logger.debug("------------当前的数据集：kmeans")
    # subprocess.call(['python', '../dataset/preprocess_atttack_strategy3_AST_kmeans_vulnerability_detection.py'])
    # main(training=True, evaluating=True, testing=True)
    # 调用执行Python脚本文件
    # 这个思路彻底失败了，说明对有缺陷的代码添加15中算子生成的代码，在DNN看起来是噪音。为什么呢，因为非但不能提升预测性能，反而降低了纯净数据的性能。
    # 有鉴于此，改变当前的思路为：
    # （1）在原始数据上跑一遍，把训练数据中错误的数据样本挑出来，我们认为这批数据是有问题的，
    # （2）从鲁棒性攻击的角度入手，使用遗传算法生成对抗样例；
    # （3）对原来的模型进行增量训练，理论上来讲，可以提升模型的泛化能力，表现出来就是测试数据集上ACC提升。
    # （4）补充问题，那么这些基于训练数据生成的对抗数据，其泛化能力有多强呢，能不能直接迁移到另外的模型上去重训练呢，比如从codebert到regevd，
    # 也就是说能不能套用薛老师他们提出的跨模型迁移的做法呢，这个做法其实是更有意义的。
    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # logger.debug("------------当前的数据集：鲁棒性攻击数据集")
    # subprocess.call(['python', '../dataset/preprocess_RobustnessAttack_OutputRawVulnerabilityCode.py'])
    # main(training=True,  evaluating=True, testing=False)

    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # print("------------当前的数据集：raw_data")
    # subprocess.call(['python', '../dataset/preprocess_atttack_strategy1_raw_data_vulnerability_detection.py'])
    # os.chdir('../code')


    # 注意，以下这行代码只是实现了模型训练。
    # main(training=True,  deletePreviousTrainedModel=True)
    # （1）在原始数据上跑一遍，把训练数据中错误的数据样本挑出来，我们认为这批数据是有问题的，
    # 切记：第一步是在模型训练完成以后，做的
    # main(training=True,evaluating=True, deletePreviousTrainedModel=True)
    # 这样做的目的只是为了生成预测对的测试样例json文件，为做鲁棒性攻击准备。
    # main(training=False, evaluating=False,
    #      testing=False,
    #      deletePreviousTrainedModel=False,
    #      isGenerateTestSuccessSamples=True,
    #      isRevalidate=False, )
    # # 接下来做第二步：（2）从鲁棒性攻击的角度入手，使用遗传算法生成对抗样例；
    #
    # # 最后一步骤是模型重训练。（3）对原来的模型进行增量训练，理论上来讲，可以提升模型的泛化能力，表现出来就是测试数据集上ACC提升。
    # 将输出的攻击结果合并，并验证最终的攻击效果
    # 这是在验证模型
    # main(training=False, evaluating=False,
    #      testing=False,
    #      deletePreviousTrainedModel=False,
    #      isGenerateTestSuccessSamples=False,
    #      isRevalidate=True, )

    # 这是在重训练模型
    main(training=False, evaluating=False,
         testing=False,
         deletePreviousTrainedModel=False,
         isGenerateTestSuccessSamples=False,
         isRevalidate=False,
         retraining=True)

    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # logger.debug("------------当前的数据集：Robustness Attack Data,策略：随机策略，RA")
    # subprocess.call(['python', '../dataset/preprocess_RobustnessAttack_Strategy_RA.py'])
    # # 注意，以下这行代码只是实现了模型训练。
    # # main(training=True,  deletePreviousTrainedModel=True)
    # # （1）在原始数据上跑一遍，把训练数据中错误的数据样本挑出来，我们认为这批数据是有问题的，
    # # 切记：第一步是在模型训练完成以后，做的
    # # main(training=True, evaluating=True, deletePreviousTrainedModel=True)
    # # main(training=False, evaluating=False, deletePreviousTrainedModel=False)
    # # 接下来做第二步：（2）从鲁棒性攻击的角度入手，使用遗传算法生成对抗样例；
    # # 最后一步骤是模型重训练。（3）对原来的模型进行增量训练，理论上来讲，可以提升模型的泛化能力，表现出来就是测试数据集上ACC提升。
    # main(training=True, evaluating=True, deletePreviousTrainedModel=True)


    # # 更改当前工作目录
    # os.chdir('../dataset/')
    # logger.debug("------------当前的数据集：Robustness Attack Data,策略：遗传算法策略，GA")
    # subprocess.call(['python', '../dataset/preprocess_RobustnessAttack_Strategy_GA_AddTrain.py'])
    # # 注意，以下这行代码只是实现了模型训练。
    # # main(training=True,  deletePreviousTrainedModel=True)
    # # （1）在原始数据上跑一遍，把训练数据中错误的数据样本挑出来，我们认为这批数据是有问题的，
    # # 切记：第一步是在模型训练完成以后，做的
    # # main(training=True, evaluating=True, deletePreviousTrainedModel=True)
    # # main(training=False, evaluating=False, deletePreviousTrainedModel=False)
    # # 接下来做第二步：（2）从鲁棒性攻击的角度入手，使用遗传算法生成对抗样例；
    # # 最后一步骤是模型重训练。（3）对原来的模型进行增量训练，理论上来讲，可以提升模型的泛化能力，表现出来就是测试数据集上ACC提升。
    # main(training=True, evaluating=True, deletePreviousTrainedModel=True)