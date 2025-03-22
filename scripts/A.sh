

python ./main.py /data/dataset/zhangchenyu/tta/test-time-adaptation/classification/data \
    --test_sets A --myclip --tpt --text_prompt_ema --text_prompt_ema_one_weight \
    --text_prompt_ema_one_weight_h=5000 --text_prompt_ema_w=0.1 --image_prompts \
    --image_prompt_ema=4 --image_prompt_ema_h=5000 --image_prompt_ema_w=0.1 \
    --info=A9/final__ --resize_flag=True \
    --resize=410 --resolution=224 --gpu 2 --n_ctx=6 \
    --ctx_init=a_low_resolution_photo_of_the_CLS,a_photo_of_the_large_CLS,itap_of_a_CLS,a_bad_photo_of_the_CLS,a_CLS_in_a_video_game,art_of_the_CLS,a_photo_of_the_small_CLS,a_dark_photo_of_a_CLS \
    --image_feature_threshold 0.65 --use_group 1 --use_retention 1 --seed 11 --batch_num 128 \
    --w_step 1.0 --w_prompt 0.1 --w_pow 1.1 --mse_loss_weight 1.0 \
    --limit 10 --degree 3 --graph_alpha 0.3 --ensemble 1 --learn_all 0 --use_ensemble 1 --ensemble_rate 0.6 --use_blip 0 --enable_aug 1

# python ./main.py /data/dataset/zhangchenyu/tta/test-time-adaptation/classification/data \
#     --test_sets A --myclip --tpt --text_prompt_ema --text_prompt_ema_one_weight \
#     --text_prompt_ema_one_weight_h=5000 --text_prompt_ema_w=0.1 --image_prompts \
#     --image_prompt_ema=4 --image_prompt_ema_h=5000 --image_prompt_ema_w=0.1 \
#     --info=A9/final__ --resize_flag=True \
#     --resize=410 --resolution=224 --gpu 2 --n_ctx=6 \
#     --ctx_init=a_low_resolution_photo_of_the_CLS,a_photo_of_the_large_CLS,itap_of_a_CLS,a_bad_photo_of_the_CLS,a_CLS_in_a_video_game,art_of_the_CLS,a_photo_of_the_small_CLS,a_dark_photo_of_a_CLS \
#     --image_feature_threshold 0.65 --use_group 1 --use_retention 1 --seed 11 --batch_num 64 \
#     --w_step 1.0 --w_prompt 0.1 --w_pow 1.1 --mse_loss_weight 1.0 --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1\
#     --limit 10 --degree 3 --graph_alpha 0.3 --ensemble 1 --learn_all 0 --use_ensemble 1 --ensemble_rate 0.6 --use_blip 0 --enable_aug 1

