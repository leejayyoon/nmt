datadir="./data/nmt_data"
train_dir="./model_dir/nmt_3layer_gnmt"
python -m nmt.nmt \
    --attention_architecture=gnmt_v2 \
    --attention=scaled_luong \
    --encoder_type=gnmt \
    --src=vi --tgt=en \
    --vocab_prefix=${train_dir}/vocab  \
    --train_prefix=${datadir}/train \
    --dev_prefix=${datadir}/tst2012  \
    --test_prefix=${datadir}/tst2013 \
    --out_dir=${train_dir} \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 
    # \
    # --metrics=bleu



# mkdir /tmp/nmt_attention_model

# python -m nmt.nmt \
#     --attention=scaled_luong \
#     --src=vi --tgt=en \
#     --vocab_prefix=/tmp/nmt_data/vocab  \
#     --train_prefix=/tmp/nmt_data/train \
#     --dev_prefix=/tmp/nmt_data/tst2012  \
#     --test_prefix=/tmp/nmt_data/tst2013 \
#     --out_dir=/tmp/nmt_attention_model \
#     --num_train_steps=12000 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=128 \
#     --dropout=0.2 \
#     --metrics=bleu


# After training, we can use the same inference command with the new out_dir for inference:

# python -m nmt.nmt \
#     --out_dir=/tmp/nmt_attention_model \
#     --inference_input_file=/tmp/my_infer_file.vi \
#     --inference_output_file=/tmp/nmt_attention_model/output_infer