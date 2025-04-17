# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from argparse import ArgumentParser
import pandas as pd
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *
import csv

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        # '--score-thr', type=float, default=0.3, help='bbox score threshold')
                     '--score-thr', type = float, default = 0.3, help = 'bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    # print(args)
    return args


def main(args):
    # config_file = 'projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py'
    # checkpoint_file = 'model/co_deformable_detr_r50_1x_coco.pth'
    config_file = '../projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
    checkpoint_file = '../model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
    # img = "/media/kewei/TRANSCEND/rain_data/oxford/05-29/left_undistort/1432894587792800.png"
    # out_file = "demo/1432892301161732.txt"
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=args.device)
    # test a single image

    # Path to the folder containing images
    # folder_path = '/home/kewei/rain_data/singapore/one/left_un_dist/'
    # folder_path = '/home/kewei/rain_data/oxford/05-29/left_undistort/'
    folder_path = '/home/kewei/rain_data/oxford/10-29/left_undistort/'

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (jpg, png, etc.)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = os.path.join(folder_path, filename)

            result = inference_detector(model, img)

            bbox_result, segm_result = result, None  # list of array
            bboxes = np.vstack(bbox_result)  # nparray
            print(bboxes)

            # image_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_path = os.path.join(os.path.dirname(image_path), image_name + ".txt")
            image_name = os.path.splitext(os.path.basename(img))[0]
            output_path = os.path.join(os.path.dirname(img), 'codetr_bb', image_name + ".txt")
            # output_path = 'aberr.txt'
            with open(output_path, 'w') as f:
                for sublist in bboxes:
                    f.write(' '.join(map(str, sublist)) + '\n')

            # test a single image
            # args.out_file = "/media/kewei/TRANSCEND/rain_data/oxford/05-29/left_undistort/1432894587792800_out.png"
            args.out_file = os.path.join(os.path.dirname(img), 'codetr_out', image_name + ".png")
            # show the results
            show_result_pyplot(
                model,
                img,
                result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=args.out_file)



async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    # args.out_file = "demo/1432892301161732_out.jpg"
    # args.out_file = "/media/kewei/TRANSCEND/rain_data/oxford/05-29/left_undistort/1432894587792800_out.png"
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
