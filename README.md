# Train the model, Train the model using 50 samples for training, 20 samples for testing, and the first 100 genes as features.
python3 train.py \
  --data_folder ./BRCA \
  --num_class 5 \
  --view 1 \
  --tr_keep 50 \
  --te_keep 20 \
  --feat_keep 100 \
  --epochs 200 \
  --hidden_dim 64 \
  --topk 10
# Run FHE inference
python3 amainfer.py --ckpt ./work_dir/best_poly_view1_sub50+20_feat100.bundle.pt
