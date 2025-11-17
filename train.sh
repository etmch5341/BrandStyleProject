python3 ./src/training/prepare_dataset.py --data_dir ./data/small_set_style1 --action validate
python src/training/train_lora.py \
  --data_dir ./data/small_set_style1 \
  --output_dir ./lora_output \
  --max_train_steps 2000 \
  --lora_rank 32 \
  --learning_rate 1e-4