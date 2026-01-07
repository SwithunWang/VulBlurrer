# test_escape_score.py
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5EncoderModel
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

codebleu_dir = os.path.join(PROJECT_ROOT, "codebleu-0.6.0")
if os.path.isdir(codebleu_dir):
    sys.path.append(codebleu_dir)
    try:
        from codebleu import calc_codebleu
    except Exception:
        calc_codebleu = None
else:
    calc_codebleu = None


def compute_similarity(ori_code, adv_code, embed_tokenizer, embed_model, device):
    inputs1 = embed_tokenizer(ori_code, return_tensors="pt", truncation=True, padding=True).to(device)
    inputs2 = embed_tokenizer(adv_code, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        emb1 = embed_model(**inputs1).last_hidden_state.mean(dim=1)
        emb2 = embed_model(**inputs2).last_hidden_state.mean(dim=1)
    
    sim = F.cosine_similarity(emb1, emb2).item()
    return 1-sim

def get_codebleu(ori_code, adv_code):
    if calc_codebleu is None:
        print("Codebleu 模块不可用，跳过 CodeBLEU 计算。")
        return 0.0
    reference_list = [ori_code]
    pred_list = [adv_code]
    codebleu_score = calc_codebleu(reference_list, pred_list, lang="c", weights=(0.1, 0.1, 0.2, 0.6), tokenizer=None)
    return codebleu_score.get('codebleu', 0.0)

def compute_fluency(adv_code: str, mlm_tokenizer, mlm_model, device, max_len=512):
    try:
        input_ids = mlm_tokenizer.encode(adv_code, return_tensors="pt")[0]
    except Exception as e:
        print(f"tokenize 失败: {e}")
        return 0.0, 0.0, 0

    fluency_scores = []

    for start_idx in range(0, len(input_ids), max_len - 2):
        end_idx = min(start_idx + max_len - 2, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        chunk_ids = torch.tensor([mlm_tokenizer.cls_token_id] + chunk_ids.tolist() + [mlm_tokenizer.sep_token_id]).unsqueeze(0).to(device)

        with torch.no_grad():
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

    return flu


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    ori_code = """
    int add(int a, int b) {
        return a + b;
    }
    """

    adv_code = """
    int add_numbers(int x, int y) {
        return x + y;
    }
    """
    result = get_codebleu(ori_code, adv_code)
    print(result)


if __name__ == "__main__":
    main()


