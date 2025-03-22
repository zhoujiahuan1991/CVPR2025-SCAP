

# python ./main.py /data/dataset/liuzichen --test_sets K  --tpt --myclip \
#     --text_prompt_ema --text_prompt_ema_one_weight --text_prompt_ema_one_weight_h=5000 \
#     --text_prompt_ema_w=0.1 --image_prompts --image_prompt_ema=4 --image_prompt_ema_h=5000 \
#     --image_prompt_ema_w=0.1 --info=K1/final__ \
#     --resize_flag=True --resize=245 --resolution=224 \
#     --ctx_init=a_sketch_photo_of_a_CLS --n_ctx=5 --gpu 4 --batch_num 32 \
#     --image_feature_threshold 0.9 --use_group 1 --use_retention 1 \
#     --reset_to_retention 1 --seed 0 \
#     --w_step 1.0 --w_prompt 0.1 --pos_alpha 1.0 --neg_alpha 0.35 --pos_beta 8.0 --neg_beta 1.0 \
#     --pos_ema_enabled 0 --mse_loss_weight 0.10 \
#     --limit 6 --degree 4 --graph_alpha 0.4

#for a in 0.9
#do
#    for b in 0.1
#    do
#
#        python ./main.py /data/dataset/liuzichen --test_sets K  --tpt --myclip \
#            --text_prompt_ema --text_prompt_ema_one_weight --text_prompt_ema_one_weight_h=5000 \
#            --text_prompt_ema_w=0.1 --image_prompts --image_prompt_ema=4 --image_prompt_ema_h=5000 \
#            --image_prompt_ema_w=0.1 --info=K1/sketch_firstthershold_${a} \
#            --resize_flag=True --resize=245 --resolution=224 \
#            --ctx_init=a_sketch_of_a_CLS,a_bad_photo_of_a_CLS,a_photo_of_many_CLS,a_sculpture_of_a_CLS,a_photo_of_the_hard_to_see_CLS,a_low_resolution_photo_of_the_CLS,a_rendering_of_a_CLS,graffiti_of_a_CLS,a_bad_photo_of_the_CLS,a_cropped_photo_of_the_CLS,a_tattoo_of_a_CLS,the_embroidered_CLS,a_photo_of_a_hard_to_see_CLS,a_bright_photo_of_a_CLS,a_photo_of_a_clean_CLS,a_photo_of_a_dirty_CLS,a_dark_photo_of_the_CLS,a_drawing_of_a_CLS,a_photo_of_my_CLS,the_plastic_CLS,a_photo_of_the_cool_CLS,a_close-up_photo_of_a_CLS,a_black_and_white_photo_of_the_CLS,a_painting_of_the_CLS,a_painting_of_a_CLS,a_pixelated_photo_of_the_CLS,a_sculpture_of_the_CLS,a_bright_photo_of_the_CLS,a_cropped_photo_of_a_CLS,a_plastic_CLS,a_photo_of_the_dirty_CLS,a_jpeg_corrupted_photo_of_a_CLS,a_blurry_photo_of_the_CLS,a_photo_of_the_CLS,a_good_photo_of_the_CLS,a_rendering_of_the_CLS,a_CLS_in_a_video_game,a_photo_of_one_CLS,a_doodle_of_a_CLS,a_close-up_photo_of_the_CLS,a_photo_of_a_CLS,the_origami_CLS,the_CLS_in_a_video_game,a_sketch_of_a_CLS,a_doodle_of_the_CLS,a_origami_CLS,a_low_resolution_photo_of_a_CLS,the_toy_CLS,a_rendition_of_the_CLS,a_photo_of_the_clean_CLS,a_photo_of_a_large_CLS,a_rendition_of_a_CLS,a_photo_of_a_nice_CLS,a_photo_of_a_weird_CLS,a_blurry_photo_of_a_CLS,a_cartoon_CLS,art_of_a_CLS,a_sketch_of_the_CLS,a_embroidered_CLS,a_pixelated_photo_of_a_CLS,itap_of_the_CLS,a_jpeg_corrupted_photo_of_the_CLS,a_good_photo_of_a_CLS,a_plushie_CLS,a_photo_of_the_nice_CLS,a_photo_of_the_small_CLS,a_photo_of_the_weird_CLS,the_cartoon_CLS,art_of_the_CLS,a_drawing_of_the_CLS,a_photo_of_the_large_CLS,a_black_and_white_photo_of_a_CLS,the_plushie_CLS,a_dark_photo_of_a_CLS,itap_of_a_CLS,graffiti_of_the_CLS,a_toy_CLS,itap_of_my_CLS,a_photo_of_a_cool_CLS,a_photo_of_a_small_CLS,a_tattoo_of_the_CLS \
#            --n_ctx=5 --gpu 3 --batch_num 16 \
#            --image_feature_threshold $a --use_group 1 --use_retention 0 \
#            --reset_to_retention 0 --seed 0 \
#            --w_step 0.0 --w_prompt 0.0 --pos_alpha 1.0 --neg_alpha 0.35 --pos_beta 8.0 --neg_beta 1.0 \
#            --pos_ema_enabled 0 --mse_loss_weight 0.10 \
#            --limit 10 --degree 4 --graph_alpha 0.4 --ensemble 1 --learn_all 0 --use_ensemble 1 --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1

#    done
#done

# below is 51.61


for a in 0.9
do
    for b in 0.1
    do

        python ./main.py /data/dataset/liuzichen --test_sets K  --tpt --myclip \
            --text_prompt_ema --text_prompt_ema_one_weight --text_prompt_ema_one_weight_h=5000 \
            --text_prompt_ema_w=0.1 --image_prompts --image_prompt_ema=4 --image_prompt_ema_h=5000 \
            --image_prompt_ema_w=0.1 --info=K1/sketch_firstthershold_${a} \
            --resize_flag=True --resize=245 --resolution=224 \
            --ctx_init=a_sketch_of_a_CLS,a_bad_photo_of_a_CLS,a_photo_of_many_CLS,a_sculpture_of_a_CLS,a_photo_of_the_hard_to_see_CLS,a_low_resolution_photo_of_the_CLS,a_rendering_of_a_CLS,graffiti_of_a_CLS,a_bad_photo_of_the_CLS,a_cropped_photo_of_the_CLS,a_tattoo_of_a_CLS,the_embroidered_CLS,a_photo_of_a_hard_to_see_CLS,a_bright_photo_of_a_CLS,a_photo_of_a_clean_CLS,a_photo_of_a_dirty_CLS,a_dark_photo_of_the_CLS,a_drawing_of_a_CLS,a_photo_of_my_CLS,the_plastic_CLS,a_photo_of_the_cool_CLS,a_close-up_photo_of_a_CLS,a_black_and_white_photo_of_the_CLS,a_painting_of_the_CLS,a_painting_of_a_CLS,a_pixelated_photo_of_the_CLS,a_sculpture_of_the_CLS,a_bright_photo_of_the_CLS,a_cropped_photo_of_a_CLS,a_plastic_CLS,a_photo_of_the_dirty_CLS,a_jpeg_corrupted_photo_of_a_CLS,a_blurry_photo_of_the_CLS,a_photo_of_the_CLS,a_good_photo_of_the_CLS,a_rendering_of_the_CLS,a_CLS_in_a_video_game,a_photo_of_one_CLS,a_doodle_of_a_CLS,a_close-up_photo_of_the_CLS,a_photo_of_a_CLS,the_origami_CLS,the_CLS_in_a_video_game,a_sketch_of_a_CLS,a_doodle_of_the_CLS,a_origami_CLS,a_low_resolution_photo_of_a_CLS,the_toy_CLS,a_rendition_of_the_CLS,a_photo_of_the_clean_CLS,a_photo_of_a_large_CLS,a_rendition_of_a_CLS,a_photo_of_a_nice_CLS,a_photo_of_a_weird_CLS,a_blurry_photo_of_a_CLS,a_cartoon_CLS,art_of_a_CLS,a_sketch_of_the_CLS,a_embroidered_CLS,a_pixelated_photo_of_a_CLS,itap_of_the_CLS,a_jpeg_corrupted_photo_of_the_CLS,a_good_photo_of_a_CLS,a_plushie_CLS,a_photo_of_the_nice_CLS,a_photo_of_the_small_CLS,a_photo_of_the_weird_CLS,the_cartoon_CLS,art_of_the_CLS,a_drawing_of_the_CLS,a_photo_of_the_large_CLS,a_black_and_white_photo_of_a_CLS,the_plushie_CLS,a_dark_photo_of_a_CLS,itap_of_a_CLS,graffiti_of_the_CLS,a_toy_CLS,itap_of_my_CLS,a_photo_of_a_cool_CLS,a_photo_of_a_small_CLS,a_tattoo_of_the_CLS \
            --n_ctx=5 --gpu 3 --batch_num 32 \
            --image_feature_threshold $a --use_group 0 --use_retention 0 \
            --reset_to_retention 0 --seed 0 \
            --w_step 0.0 --w_prompt 0.0 --pos_alpha 1.0 --neg_alpha 0.35 --pos_beta 8.0 --neg_beta 1.0 \
            --pos_ema_enabled 0 --mse_loss_weight 0.10 \
            --limit 10 --degree 4 --graph_alpha 0.4 --ensemble 1 --learn_all 0 --use_ensemble 1 --pos_enabled 1 --neg_enabled 1 --enable1 1 --enable2 1

    done
done
