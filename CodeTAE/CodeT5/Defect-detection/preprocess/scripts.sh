CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
    --store_path ./dataset/test_success_data_CodeT5_BigVul_subs_k20.jsonl \
    --base_model= \
    --eval_data_file=\
    --block_size 512


 CUDA_VISIBLE_DEVICES=0 python get_substitutes_gan.py \
    --store_path ./dataset/test_success_data_CodeT5_BigVul_subs_gan_k60.jsonl \
    --base_model  \
    --eval_data_file  \
    --block_size 512