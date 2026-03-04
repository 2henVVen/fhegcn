# Train the model
python3 train.py \
--data_folder ./BRCA \
--num_class 5 \
--view 1 \
--tr_keep 50 \
--te_keep 20 \
--feat_keep 100

# Run FHE inference
python3 amainfer.py --ckpt ./work_dir/best_poly_view1_sub50+20_feat100.bundle.pt
