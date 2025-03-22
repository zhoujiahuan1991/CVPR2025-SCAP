
python ./main.py /data/dataset/zhangchenyu/tta/test-time-adaptation/classification/data \
    --test_sets V --myclip --tpt --text_prompt_ema --text_prompt_ema_one_weight \
    --text_prompt_ema_one_weight_h=5000 --text_prompt_ema_w=0.1 --image_prompts \
    --image_prompt_ema=4 --image_prompt_ema_h=5000 --image_prompt_ema_w=0.1 \
    --info=V2/final- --resize_flag=True \
    --resize=256 --resolution=224 --gpu 2 \
    --ctx_init=a_low_resolution_photo_of_a_CLS,a_photo_of_the_large_CLS,itap_of_a_CLS,a_bad_photo_of_the_CLS,a_CLS_in_a_video_game,a_photo_of_the_hard_to_see_CLS,a_photo_of_the_small_CLS,a_photo_of_many_CLS \
    --n_ctx=6 \
    --image_feature_threshold 0.8 --use_group 0 --use_retention 0 --seed 19 --batch_num 32 \
    --w_step 0.0 --w_prompt 0.0 --w_pow 1.1 --mse_loss_weight 0.10 \
    --limit 10 --degree 4 --graph_alpha 0.4 --save_groups 0\
    --pos_alpha 2.0 --neg_alpha 0.117 --pos_beta 8.0 --neg_beta 1.0 --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1 \
    --ensemble 1 --learn_all 0 --use_ensemble 1 --ensemble_rate 0.5 --simam 0\

# for a in 5000
# do
#     for b in 0.5
#     do   
#         python ./main.py /data/dataset/zhangchenyu/tta/test-time-adaptation/classification/data \
#             --test_sets V --myclip --tpt --text_prompt_ema --text_prompt_ema_one_weight \
#             --text_prompt_ema_one_weight_h=5000 --text_prompt_ema_w=0.1 --image_prompts \
#             --image_prompt_ema=4 --image_prompt_ema_h=5000 --image_prompt_ema_w=0.1 \
#             --info=V2/h${a}_rates${b}- --resize_flag=True \
#             --resize=256 --resolution=224 --gpu 3 \
#             --ctx_init=a_low_resolution_photo_of_a_CLS,a_photo_of_the_large_CLS,itap_of_a_CLS,a_bad_photo_of_the_CLS,a_CLS_in_a_video_game,a_photo_of_the_hard_to_see_CLS,a_photo_of_the_small_CLS,a_photo_of_many_CLS \
#             --n_ctx=6 \
#             --image_feature_threshold 0.8 --use_group 0 --use_retention 0 --seed 19 --batch_num 32 \
#             --w_step 0.0 --w_prompt 0.0 --w_pow 1.1 --mse_loss_weight 0.10 \
#             --limit 10 --degree 4 --graph_alpha 0.4 --save_groups 0\
#             --pos_alpha 2.0 --neg_alpha 0.117 --pos_beta 8.0 --neg_beta 1.0 --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1 \
#             --ensemble 1 --learn_all 0 --use_ensemble 1 --ensemble_rate $b --simam 1  --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1\
            
#     done
# done
