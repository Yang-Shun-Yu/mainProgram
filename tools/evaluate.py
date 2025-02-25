import sys
sys.path.append('/home/eddy/Desktop/MasterThesis/mainProgram/tools/py_motmetrics')

import os
import argparse
import py_motmetrics.motmetrics as mm

from loguru import logger

def evaluate(gt_dir, ts_dir, mode):
    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()

    if mode == 'multi_cam':
        accs = []
        names = []

        gt_files = sorted(os.listdir(gt_dir))
        ts_files = sorted(os.listdir(ts_dir))

        for gt_file, ts_file in zip(gt_files, ts_files):
            gt_path = os.path.join(gt_dir, gt_file)
            ts_path = os.path.join(ts_dir, ts_file)

            # compare the same title files
            if os.path.splitext(gt_file)[0] == os.path.splitext(ts_file)[0]:
                gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(ts_path, fmt="mot15-2D")
                names.append(os.path.splitext(os.path.basename(ts_path))[0])
                accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)) # ground truth 覆蓋率 > 0.5 才是對的

        summary = mh.compute_many(accs, metrics=metrics, generate_overall=True)
        print("Score for total file: IDF1 ", summary.idf1.OVERALL, " + MOTA ", summary.mota.OVERALL, " = ", summary.idf1.OVERALL+summary.mota.OVERALL)
        # 完整的 motmetrics 各項評分成果 V
        logger.info(f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')

    elif mode == 'single_cam':
        gt_files = sorted(os.listdir(gt_dir))
        ts_files = sorted(os.listdir(ts_dir))

        for gt_file, ts_file in zip(gt_files, ts_files):
            gt_path = os.path.join(gt_dir, gt_file)
            ts_path = os.path.join(ts_dir, ts_file)

            # compare the same title files
            if os.path.splitext(gt_file)[0] == os.path.splitext(ts_file)[0]:
                gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(ts_path, fmt="mot15-2D")
                name = os.path.splitext(os.path.basename(ts_path))[0]
                acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5) # ground truth 覆蓋率 > 0.5 才是對的

                summary = mh.compute_many([acc], metrics=metrics, generate_overall=True)
                print(f"Score for {name}: IDF1 {summary.idf1.OVERALL}, MOTA {summary.mota.OVERALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate multiple object tracking results.')
    parser.add_argument('--gt_dir', type=str, help='Path to the ground truth directory')
    parser.add_argument('--ts_dir', type=str, help='Path to the tracking result directory')
    parser.add_argument('--mode', type=str, choices=['multi_cam', 'single_cam'], default='multi_cam', help='Evaluation mode')

    args = parser.parse_args()

    evaluate(args.gt_dir, args.ts_dir, args.mode)
