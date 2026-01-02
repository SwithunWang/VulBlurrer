import torch
import os
import json
import random
import numpy as np
import sys
import logging
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 新增/修改：把 stdout 的输出同时写入新的日志文件（每次运行创建新文件），以确保日志不会丢失
try:
    log_dir = os.path.dirname(__file__)
    # 使用时间戳创建唯一的日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    auto_log_path = os.path.join(log_dir, f"eval_run_codeT5_take2{timestamp}.log")
    # 使用覆盖模式，每次运行创建新文件
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filename=auto_log_path, filemode='w')
    _orig_stdout = sys.stdout
    class _LoggerTee:
        def write(self, msg):
            try:
                _orig_stdout.write(msg)
            except Exception:
                pass
            try:
                # 只把非空白行写入 logging，避免过多空行
                if msg and msg.strip():
                    logging.info(msg.strip())
            except Exception:
                pass
        def flush(self):
            try:
                _orig_stdout.flush()
            except Exception:
                pass
    sys.stdout = _LoggerTee()
    # 同时也把 stderr 指向 stdout，这样 print 和异常都会被记录
    sys.stderr = sys.stdout
    print(f"日志文件: {auto_log_path}")
except Exception as e:
    # 如果任何事情失败了，不要阻止脚本运行 — 继续使用原有 stdout/stderr
    print(f"日志设置失败: {e}")
    pass

from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
from typing import Tuple

# Path to codebleu (adjustable via CLI)
codebleu_dir = ""
if os.path.isdir(codebleu_dir):
    sys.path.append(codebleu_dir)
    try:
        from codebleu import calc_codebleu
    except Exception:
        calc_codebleu = None
else:
    calc_codebleu = None


def get_codebleu(ori_code, adv_code):
    if calc_codebleu is None:
        # If codebleu not available, return 0.0 and warn
        print("警告: codebleu 模块不可用，跳过 CodeBLEU 计算。")
        return 0.0

    reference_list = [ori_code]
    pred_list = [adv_code]
    codebleu_score = calc_codebleu(reference_list, pred_list, lang="c", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return codebleu_score.get('codebleu', 0.0)


def compute_change_ratio(original_code: str, adv_code: str) -> float:
    """
    计算两个代码的 change ratio (修改比例)
    比例 = 不同字符数 / 原始代码长度
    """
    len_orig = len(original_code)
    if len_orig == 0:
        return 0.0

    # 逐字符比较
    diff_count = sum(1 for o, a in zip(original_code, adv_code) if o != a)
    # 如果 adv_code 比 original_code 长，算作额外的修改
    diff_count += abs(len(adv_code) - len(original_code))

    change_ratio = diff_count / len_orig
    return change_ratio


# ------------------ 固定随机种子 ------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # These flags affect reproducibility and performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------ 计算流畅度 ------------------
def compute_fluency(adv_code: str, mlm_tokenizer, mlm_model, device, max_len=512) -> Tuple[float, float, int]:
    """Compute fluency by masking each token and taking the token probability assigned to the true token.
    Returns (average_prob, sum_prob, token_count).
    Note: this is relatively slow because it runs the model per-token; keep that in mind for large datasets.
    """
    # Tokenize and get input ids (flatten)
    try:
        input_ids = mlm_tokenizer.encode(adv_code, return_tensors="pt")[0]
    except Exception as e:
        print(f"tokenize 失败: {e}")
        return 0.0, 0.0, 0

    fluency_scores = []

    # 按 max_len 切块处理（为 MLM 增加 CLS/SEP）
    for start_idx in range(0, len(input_ids), max_len - 2):
        end_idx = min(start_idx + max_len - 2, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        chunk_ids = torch.tensor([mlm_tokenizer.cls_token_id] + chunk_ids.tolist() + [mlm_tokenizer.sep_token_id]).unsqueeze(0).to(device)

        with torch.no_grad():
            # 对每个位置做一次 mask（跳过 CLS/SEP）
            for i in range(1, chunk_ids.size(1) - 1):
                masked_input = chunk_ids.clone()
                masked_input[0, i] = mlm_tokenizer.mask_token_id

                outputs = mlm_model(masked_input)
                logits = outputs.logits[0, i]
                true_id = chunk_ids[0, i]

                prob = torch.softmax(logits, dim=-1)[true_id]
                fluency_scores.append(prob.item())

    if not fluency_scores:
        return 0.0, 0.0, 0

    flu = float(sum(fluency_scores) / len(fluency_scores))
    sum_flu = float(sum(fluency_scores))

    return flu, sum_flu, len(fluency_scores)


# ------------------ 主程序 ------------------

def save_atomic(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial code samples with fluency, CodeBLEU and change-ratio.")
    parser.add_argument("--input", "-i", default="", help="Input JSONL file with adversarial samples")
    parser.add_argument("--output", "-o", default="", help="Output JSON file for results")
    parser.add_argument("--model", "-m", default="microsoft/codebert-base", help="Masked LM model for fluency scoring")
    args = parser.parse_args()

    jsonl_file = args.input
    output_file = args.output
    codebert_model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer/model {codebert_model_name} on {device}...")
    mlm_tokenizer = AutoTokenizer.from_pretrained(codebert_model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(codebert_model_name).to(device)
    mlm_model.eval()

    results = []

    # 不加载已有输出，也不跳过任何已处理的样本 —— 始终从头开始处理所有输入
    results = []

    if not os.path.isfile(jsonl_file):
        print(f"输入文件不存在: {jsonl_file}")
        return

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except Exception as e:
                print(f"第 {line_num} 行不是合法 JSON: {e}")
                continue

            idx = data.get('idx', f'unknown_{line_num}')

            orig_code = data.get('ori_func', '')
            adv_code = data.get('func', '')

            if not adv_code:
                print(f"警告: 第 {line_num} 行样本 {idx} 没有 'adv_code' 字段，跳过")
                continue

            print(f"处理样本 {idx} (行 {line_num})...")

            try:
                flu, sum_flu, len_flu = compute_fluency(adv_code, mlm_tokenizer, mlm_model, device)
            except Exception as e:
                print(f"计算流畅度时出错 (样本 {idx}): {e}")
                flu, sum_flu, len_flu = 0.0, 0.0, 0

            # Compute CodeBLEU if original code exists
            codebleu_score = 0.0
            if orig_code:
                try:
                    codebleu_score = get_codebleu(orig_code, adv_code)
                except Exception as e:
                    print(f"计算 CodeBLEU 时出错 (样本 {idx}): {e}")
                    codebleu_score = 0.0
            else:
                print(f"警告: 样本 {idx} 没有原始代码用于计算 CodeBLEU")

            # Compute change ratio
            change_ratio = 0.0
            if orig_code:
                try:
                    change_ratio = compute_change_ratio(orig_code, adv_code)
                except Exception as e:
                    print(f"计算 Change Ratio 时出错 (样本 {idx}): {e}")
                    change_ratio = 0.0
            else:
                print(f"警告: 样本 {idx} 没有原始代码用于计算 Change Ratio")

            result = {
                "idx": idx,
                "fluency_score": float(flu),
                "sum_fluency": float(sum_flu),
                "fluency_tokens": int(len_flu),
                "codebleu": float(codebleu_score),
                "change_ratio": float(change_ratio)
            }

            results.append(result)
            # 增量保存（原子替换）以支持中断恢复
            try:
                save_atomic(output_file, results)
            except Exception as e:
                print(f"写入输出文件失败 (样本 {idx}): {e}")

            print(f"样本 {idx} 处理完成: fluency={flu:.6f}, sum_flu={sum_flu:.6f}, tokens={len_flu}, codebleu={codebleu_score:.6f}, change_ratio={change_ratio:.6f}")

    # 计算平均指标
    if results:
        avg_fluency = sum(r['fluency_score'] for r in results) / len(results)
        avg_sum_fluency = sum(r['sum_fluency'] for r in results) / len(results)
        avg_fluency_tokens = sum(r['fluency_tokens'] for r in results) / len(results)
        avg_codebleu = sum(r['codebleu'] for r in results) / len(results)
        avg_change_ratio = sum(r['change_ratio'] for r in results) / len(results)
        
        print(f"\n平均指标:")
        print(f"平均 Fluency Score: {avg_fluency:.6f}")
        print(f"平均 Sum Fluency: {avg_sum_fluency:.6f}")
        print(f"平均 Fluency Tokens: {avg_fluency_tokens:.2f}")
        print(f"平均 CodeBLEU: {avg_codebleu:.6f}")
        print(f"平均 Change Ratio: {avg_change_ratio:.6f}")
        
        # 将平均指标添加到结果中一起保存
        avg_metrics = {
            "avg_fluency_score": float(avg_fluency),
            "avg_sum_fluency": float(avg_sum_fluency),
            "avg_fluency_tokens": float(avg_fluency_tokens),
            "avg_codebleu": float(avg_codebleu),
            "avg_change_ratio": float(avg_change_ratio)
        }
        
        # 在保存的结果中添加平均指标
        results_with_avg = results + [{"average_metrics": avg_metrics}]
        try:
            save_atomic(output_file, results_with_avg)
        except Exception as e:
            print(f"写入包含平均指标的输出文件失败: {e}")
    else:
        print("没有有效的结果用于计算平均指标")

    # 最终消息
    print(f"\n评估脚本执行完毕或已处理完输入文件。当前已保存结果到: {output_file}")
    print(f"共记录 {len(results)} 个样本的评估结果")


if __name__ == '__main__':
    main()