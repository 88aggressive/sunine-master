#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np
import os

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="voxceleb1")
#     parser.add_argument('--src_trl_path', help='src_trials_path', type=str, default="voxceleb1_test_v2.txt")
#     parser.add_argument('--dst_trl_path', help='dst_trials_path', type=str, default="new_trials.lst")
#     parser.add_argument('--apply_vad', action='store_true', default=False)
#     args = parser.parse_args()
#
#     trials = np.loadtxt(args.src_trl_path, dtype=str)
#
#     f = open(args.dst_trl_path, "a+")
#     for item in trials:
#         enroll_path = os.path.join(args.voxceleb1_root, item[1])
#         test_path = os.path.join(args.voxceleb1_root, item[2])
#         if args.apply_vad:
#             enroll_path = enroll_path.strip("*.wav") + ".vad"
#             test_path = test_path.strip("*.wav") + ".vad"
#         f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="voxceleb1")
    parser.add_argument('--src_trl_path', help='src_trials_path', type=str, default="voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trl_path', help='dst_trials_path', type=str, default="new_trials.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    # 加载测试数据列表
    trials = np.loadtxt(args.src_trl_path, dtype=str)

    # 打开保存新的数据列表的文件
    f = open(args.dst_trl_path, "a+")

    # 遍历测试数据列表
    for item in trials:
        # 获取注册和测试语音的路径
        enroll_path = os.path.join(args.voxceleb1_root, item[1])
        test_path = os.path.join(args.voxceleb1_root, item[2])
        # 如果使用VAD，则对路径进行修改
        if args.apply_vad:
            enroll_path = enroll_path.strip("*.wav") + ".vad"
            test_path = test_path.strip("*.wav") + ".vad"
        # 将新的测试数据保存到文件
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

