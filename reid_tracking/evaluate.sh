#!/bin/bash

# 定义目录路径
TS_RESULTS_DIR='/home/eddy/Desktop/MasterThesis/mainProgram/ts_results'
GT_RESULT_DIR='/home/eddy/Desktop/MasterThesis/mainProgram/gt_results'

SOURCE_LABELS='/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/multi_camera_labels/'
TARGET_LABELS='/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels/'
cp -r "${SOURCE_LABELS}"* "${TARGET_LABELS}"

# 清空 ts_results 目录
if [ -d "$TS_RESULTS_DIR" ]; then
    rm -rf "$TS_RESULTS_DIR"/*
    echo "已清空目录：$TS_RESULTS_DIR"
else
    echo "目录不存在：$TS_RESULTS_DIR"
    exit 1
fi

# 清空 gt_result 目录
if [ -d "$GT_RESULT_DIR" ]; then
    rm -rf "$GT_RESULT_DIR"/*
    echo "已清空目录：$GT_RESULT_DIR"
else
    echo "目录不存在：$GT_RESULT_DIR"
    exit 1
fi

# 运行第一个 Python 脚本
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/datasets/AICUP_to_MOT15.py \
    --AICUP_dir '/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels' \
    --MOT15_dir "$TS_RESULTS_DIR"

# 检查上一个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "第一个脚本执行失败，终止后续操作。"
    exit 1
fi

# 运行第二个 Python 脚本
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/datasets/AICUP_to_MOT15.py \
    --AICUP_dir '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels' \
    --MOT15_dir "$GT_RESULT_DIR"

# 检查上一个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "第二个脚本执行失败，终止后续操作。"
    exit 1
fi

# 运行第三个 Python 脚本
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/evaluate.py \
    --gt_dir "$GT_RESULT_DIR/" \
    --ts_dir "$TS_RESULTS_DIR/"

# 检查上一个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "第三个脚本执行失败。"
    exit 1
fi

echo "所有脚本已成功执行。"
