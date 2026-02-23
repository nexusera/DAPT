请在服务器的 DAPT 目录下依次运行以下三组命令：

1. 处理训练集 (Train)
python fetch_and_merge_baidu_ocr.py \
  --anno_json /data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_train.json \
  --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
  --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
  --extra_root2 /data/ocean/medstruct_benchmark/ocr_results \
  --output biaozhu_with_ocr/real_train_with_ocr.json \
  --use_local_baidu --local_mode accurate \
  --sleep 0.5

  
2. 处理测试集 (Test)
python fetch_and_merge_baidu_ocr.py \
  --anno_json /data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_test.json \
  --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
  --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
  --extra_root2 /data/ocean/medstruct_benchmark/ocr_results \
  --output biaozhu_with_ocr/real_test_with_ocr.json \
  --use_local_baidu --local_mode accurate \
  --sleep 0.5
3. 计算噪声特征 (Prepare Noise)
python compute_noise_from_ocr.py \
  --inputs biaozhu_with_ocr/real_train_with_ocr.json biaozhu_with_ocr/real_test_with_ocr.json \
  --output_dir biaozhu_with_ocr_noise_prepared