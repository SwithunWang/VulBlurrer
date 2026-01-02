    python get_substitutes.py \
    --store_path ./dataset/test_success_data_CodeBert_BigVul_subs_k10.jsonl \
    --base_model= \
    --eval_data_file=./dataset/test_success_data_CodeBert_BigVul.jsonl \
    --block_size 512


CUDA_VISIBLE_DEVICES=0 python get_substitutes_gan.py \
    --store_path ./dataset/test_success_data_CodeBert_BigVul_subs_gan_test.jsonl \
    --base_model= \
    --eval_data_file=./dataset/test_success_data_CodeBert_BigVul.jsonl \
    --block_size 512