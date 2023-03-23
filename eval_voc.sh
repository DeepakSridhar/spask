# #!/bin/bash

# # Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
# # Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# GPU=$1
# MODEL=$2
# #CUDA_VISIBLE_DEVICES=$GPU python evaluate_cub.py --crf --save-viz --dataset cub --restore-from umr/model_60000.pth --save-dir results/cub/ITER_60000/val/ --split val --num-parts 4
# CUDA_VISIBLE_DEVICES=$GPU python evaluate_cub.py --crf --save-viz --dataset cub --model $MODEL --restore-from snapshots_Cub/SCOPS_K5_cub/model_80000.pth --save-dir results/SCOPS_K5_cub/ITER_80000/val/ --split val --num-parts 5

# echo "Evaluating landmarks"

# CMD_EVAL="python evaluation/face_evaluation_wild.py $SAVE_DIR/$method/ITER_$iter | tee $SAVE_DIR/$method/ITER_$iter/lm_evaluation.txt"


DATASET="VOC"
DATA_DIR="/data8/VOCdevkit/VOC2012/"
DATA_LIST="data/voc/"
MODEL_DIR="snapshots_VOC_8_finetune"
TRAIN_LIST="data/voc/"
TEST_LIST="data/voc/"
MODEL="DeepLab50_2branch_Geometric"

nvidia-smi
read -e -p "Which GPU to use? : " -i "0" GPU

read -e -p "Model: "                    -i $MODEL MODEL

read -e -p "Evaluate on train set and calculate landmark errors? (y/n) : " -i "n" EVAL_TRAIN

read -e -p "Model dir: "                -i $MODEL_DIR MODEL_DIR
ls $MODEL_DIR
echo ""
read -e -p "Model name(s): "             -i "SCOPS_K8_train_VOC_finetune"  METHOD

ls $MODEL_DIR/$METHOD/model*
echo ""
read -e -p "Iter(s): "                     -i "160000" ITER


read -e -p "Save Dir: "                 -i "results_VOC" SAVE_DIR

read -e -p "extra args for test set : " -i "--crf --save-viz --input-size 128,128" ARG_TEST
read -e -p "extra args for train set : " -i "--crf --save-viz --input-size 128,128" ARG_TRAIN


for iter in $ITER
do

    for method in $METHOD
    do
        K_index=${method##*_K}
        K_index=${K_index%%_*}

        NUM_PARTS=$((K_index))
        echo 'num classes' $NUM_CLASS

        SNAPSHOT="${MODEL_DIR}/$method/model_${iter}.pth"
        # Testing
        CMD_TEST="CUDA_VISIBLE_DEVICES=${GPU} python evaluate_voc.py $ARG_TEST --dataset $DATASET --data-dir $DATA_DIR --data-list $TEST_LIST --num-parts $NUM_PARTS --model $MODEL --restore-from $SNAPSHOT --split val --save-dir $SAVE_DIR/$method/ITER_$iter/test/" 
        echo ""
        echo "$CMD_TEST"
        echo ""
        eval "$CMD_TEST"


        if [[ $EVAL_TRAIN == "y" ]]
        then
            # if [ ! -d "$SAVE_DIR/$method/ITER_$iter/train/" ]; then
            CMD_TRAIN="CUDA_VISIBLE_DEVICES=${GPU} python evaluate_voc.py $ARG_TRAIN --dataset $DATASET --data-dir $DATA_DIR --data-list $TRAIN_LIST --num-parts $NUM_PARTS --model $MODEL --restore-from $SNAPSHOT --split train --save-dir $SAVE_DIR/$method/ITER_$iter/train/"

            echo ""
            echo "$CMD_TRAIN"
            echo ""
            eval "$CMD_TRAIN"
            # fi

            echo "Evaluating landmarks"

            CMD_EVAL="python evaluation/face_evaluation_wild.py $SAVE_DIR/$method/ITER_$iter | tee $SAVE_DIR/$method/ITER_$iter/lm_evaluation.txt"
        fi

        echo ""
        echo "$CMD_EVAL"
        echo ""
        eval "$CMD_EVAL"

        #CMD_WEB="python web_visualize.py -o $SAVE_DIR/$method/ITER_$iter/web_html -dirs ./data/CelebA/img_celeba $SAVE_DIR/$method/ITER_$iter/test/landmarks $SAVE_DIR/$method/ITER_$iter/test/part_overlay $SAVE_DIR/$method/ITER_$iter/test/part_dcrf_overlay $SAVE_DIR/$method/ITER_$iter/test/part_map $SAVE_DIR/$method/ITER_$iter/test/part_map_dcrf -names Img Landmarks PartoOverlay PartoOverlayDCRF PartMaps PartMapsDCRF -ref 1"

        # echo ""
        # echo "$CMD_WEB"
        # echo ""
        # eval "$CMD_WEB"
    done
done
