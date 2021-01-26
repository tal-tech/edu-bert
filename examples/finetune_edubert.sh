export MODEL_PATH=../../path_to_EduBERT # 如果需要加载tensorflow模型，模型文件夹名需要以.cpkt结尾，如下面一行所示
#export MODEL_PATH=../edu-bert.ckpt #加载tensorflow模型的文件夹名称样例
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
