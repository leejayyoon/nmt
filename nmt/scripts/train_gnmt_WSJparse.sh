datadir="./data/nmt_data"
train_dir="./model_dir/nmt_3layer_gnmt"
python -m nmt.nmt \
    --attention_architecture=gnmt_v2 \
    --attention=scaled_luong \
    --encoder_type=gnmt \
    --src=encode --tgt=decode \
    --vocab_prefix=${train_dir}/vocab  \
    --train_prefix=${datadir}/split-train \
    --dev_prefix=${datadir}/split-dev  \
    --test_prefix=${datadir}/split-test \
    --out_dir=${train_dir} \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 
    # \
    # --metrics=bleu