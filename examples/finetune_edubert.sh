export MODEL_PATH=../../path_to_EduBERT
export DATA_DIR=../data/
export TASK_NAME=CoLA
python ../src/finetune.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --data_dir ${DATA_DIR} \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./output \
  --overwrite_output_dir \
