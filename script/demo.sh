#!/bin/sh
gpu_id=0
confidence=0.4
fore_threshold=1
move_threshold=0
output_dir="/zbs_result/"

Para1="blizzard, snowFall, wetSnow, office, pedestrians, boulevard, boats, canoe, fountain02, overpass, abandonedBox, continuousPan, zoomInZoomOut, backdoor, bungalows, cubicle, peopleInShade, corridor, library"
Para2="park"
Para3="skating, highway, PETS2006, badminton, sidewalk, traffic, fall, fountain01, parking, sofa, streetLight, tramstop, winterDriveway, tramCrossroad_1fps, tunnelExit_0_35fps, turnpike_0_5fps, bridgeEntry, streetCornerAtNight, winterStreet, intermittentPan, twoPositionPTZCam, busStation, copyMachine, diningRoom, lakeSide, turbulence3"
Para4="port_0_17fps, busyBoulvard, fluidHighway, tramStation, turbulence0, turbulence1, turbulence2"


if [ "$1" == "cdnet" ] ; then
    for data_dir in `ls -d ../cdnet2014/*/*`
    do
        subcategory=$(basename "$data_dir")
        if [[ $Para1 =~ $subcategory ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
                --output-dir ${output_dir} \
                --confidence-threshold 0.3 \
                --input-dir ${data_dir}/ \
                --noiof \
                --fore_threshold 0.3 \
                --iou_threshold 0.3 \
                --max_age 100 \
                --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                --vocabulary lvis \
                --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
        elif [[ $Para2 =~ $subcategory ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
                --output-dir ${output_dir} \
                --confidence-threshold 0.4 \
                --input-dir ${data_dir}/ \
                --noiof \
                --fore_threshold 0.3 \
                --iou_threshold 0.3 \
                --max_age 100 \
                --delta_conf 0.1 \
                --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                --vocabulary lvis \
                --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
        elif [[ $Para3 =~ $subcategory ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
                --output-dir ${output_dir} \
                --confidence-threshold 0.4 \
                --input-dir ${data_dir} \
                --fore_threshold 1 \
                --move_threshold 0 \
                --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                --vocabulary lvis \
                --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
        elif [[ $Para4 =~ $subcategory ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
                --output-dir ${output_dir} \
                --input-dir ${data_dir}/ \
                --pixel \
                --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                --vocabulary lvis \
                --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
        fi
    done
elif [ "$1" == "test" ] ; then
    # 测试
    python test.py --sub_dir ${output_dir}
elif [ "$1" == "video" ] ; then
    # 视频训练
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
        --output-dir $3 \
        --confidence-threshold ${confidence} \
        --video-input $2 \
        --fore_threshold ${fore_threshold} \
        --move_threshold ${move_threshold} \
        --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --vocabulary lvis \
        --white_list 'bus_(vehicle),truck,train_(railroad_vehicle),wheel,headlight,ladder,crossbar,windshield_wiper,flowerpot,license_plate,streetlight,street_sign,grill,lamppost,trash_can,pole' \
        --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi

# # 1109_cls_iob_bg_c.4.6_mt0_st.4_bg1.0_hit5_age20_new
# CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
#         --output-dir ${output_dir} \
#         --confidence-threshold 0.4 \
#         --input-dir ${data_dir} \
#         --fore_threshold 1 \
#         --move_threshold 0 \
#         --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
#         --vocabulary lvis \
#         --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# # 0727_c.3.5_st.3_bg.3_hit5_age100
# CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
#         --output-dir ${output_dir} \
#         --confidence-threshold 0.3 \
#         --input-dir ${data_dir}/ \
#         --noiof \
#         --fore_threshold 0.3 \
#         --iou_threshold 0.3 \
#         --max_age 100 \
#         --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
#         --vocabulary lvis \
#         --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# # 0727_c.4.5_st.3_bg.3_hit5_age100
# CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
#         --output-dir ${output_dir} \
#         --confidence-threshold 0.4 \
#         --input-dir ${data_dir}/ \
#         --noiof \
#         --fore_threshold 0.3 \
#         --iou_threshold 0.3 \
#         --max_age 100 \
#         --delta_conf 0.1 \
#         --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
#         --vocabulary lvis \
#         --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# # 0812_vibe_ns30_mm2_r10_rs16_ft50
# CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
#         --output-dir ${output_dir} \
#         --input-dir ${data_dir}/ \
#         --pixel \
#         --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
#         --vocabulary lvis \
#         --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth