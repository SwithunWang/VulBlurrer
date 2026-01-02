    python3 get_substitutes.py \
    --store_path ./dataset/test_success_data_UniXcoder_BigVul_subs_k10.jsonl \
    --base_model= \
    --eval_data_file=./dataset/test_success_data.jsonl \
    --block_size 512


CUDA_VISIBLE_DEVICES=0 python3 get_substitutes_gan.py \
    --store_path ./dataset/test_success_data_Unixcoder_BigVul_subs_gan_test.jsonl \
    --base_model= \
    --eval_data_file=./dataset/test_success_data.jsonl \
    --block_size 512