
# Copyright (c) OpenMMLab. All rights reserved.
import os
import pdb

from mmdet.apis import ( inference_detector,
                        init_detector, show_result_pyplot)
import cv2
from projects import *
import csv
from ocsort import OCSort
from trackers.ocsort_tracker.ocsort import OCSort
# from trackers.tracking_utils.timer import Timer


# # Co-detr Detection
def detection(model, img_path, image_name):
    results = []
    bboxes_list = []
    classnames_list = []
    scores_list =[]
    # Run inference
    result = inference_detector(model, img_path)
    # Extract and print class names, bounding boxes, and scores
    bbox_result, _ = result, None  # result[0] contains bbox predictions for each class
    img = cv2.imread(img_path)
    # height, width = img.shape[:2]

    # Iterate over all classes
    for class_id, bboxes in enumerate(bbox_result):
        # Skip if there are no detections for the class
        if bboxes.shape[0] == 0:
            continue

        # bboxes = np.array(bboxes)
        # Filter out the confidence level
        # threshold = 0.5
        threshold = 0.3 # 07042025 don't filter the confidence level
        bboxes = bboxes[bboxes[:, 4] > threshold]
        # Get class name
        class_name = model.CLASSES[class_id]
        if class_name in {'car', 'bus', 'truck'}:
        # if class_name in {'car', 'bus', 'truck', 'person'}:
            # Iterate over each detected bounding box for the class
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                if score >= threshold:  # Apply score threshold
                    print(
                        # f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img:{image_name} , frame id: {frame_id}")
                        f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img:{image_name}")
                    # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 100, 0), thickness=1)
                    # cv2.putText(img, f"{class_name}: {score:.2f}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    #             (36, 180, 120), 1)

                    bbox = bbox.reshape((1, 5))
                    #bboxes, class_name, score
                    # bbox_array = np.array([[x1, y1, x2, y2]])
                    # results.append((bbox_array, class_name, score))
                    bboxes_list.append(bbox[:4])  # Append only the coordinates (x1, y1, x2, y2)
                    classnames_list.append(class_name)  # Append the class name
                    scores_list.append(score)

    return np.array(bboxes_list), classnames_list, scores_list
    # return results

def main():
    # 1. Load Object Detection Model - Co-Detr
    config_file = 'projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
    checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 2. Load Object Tracking - OC-Sort
    oc_sort = OCSort(det_thresh=0.4)

    # Path to the folder containing images
    # folder_path = '/home/kewei/rain_data/singapore/one/left_un_dist/'
    folder_path = '/home/kewei/rain_data/oxford/10-29/left_undistort/'
    # folder_path = '/home/kewei/Co-DETR/images/oxford'

    frame_id = 0
    results = []
    debug_mode = True

    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        # pdb.set_trace()
        if image_name == '1446121165943091' and debug_mode:  #1446121165943091
            print(f"Found, stopping the loop.")
            debug_mode = False
        out_file = os.path.join(os.path.dirname(img_path), 'codetr_ocsort', image_name + ".png")
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        frame_id = frame_id +1

        # 3. Detect Objects
        # bboxes, class_name, scores= detection(model, img_path, image_name)
        # print(bboxes, class_name, scores)
        results = detection(model, img_path, image_name) # results [0]: bounding box, results [1]: class_name, results[2]: score
        # print ("bbounding box:", results[0])
        # result[0] will have shape (4, 1, 5)
        array = results[0]
        # To reshape this to the desired format (4, 5)
        reshaped_array = array.reshape(-1, 5)

        # 4. Tracking Objects
        tracks = oc_sort.update(reshaped_array, [height, width], [height, width])
        # tracks = oc_sort.update(bboxes, class_name, frame_id)
        print('tracks', tracks)
        for track in tracks:
            (x1,y1,x2,y2, object_id ) = np.array(track)
            print(x1,y1,x2,y2, object_id)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 100, 0), thickness=1)
            cv2.putText(img, f"{object_id}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (36, 180, 120), 1)
            # # Here, you can pass the `bboxes` to OC-SORT for tracking
            # tracks = oc_sort.update(bbox, [height, width], [height, width])
            # # tracks = oc_sort.update(bbox, class_name, frame_id)
            # online_tlwhs = []
            # online_ids = []
            # print("tracks:", tracks)
            # for t in tracks:
            #     print("t", t)
            #     tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            #     tid = t[4]
            #     vertical = tlwh[2] / tlwh[3] > 1.6 #args.aspect_ratio_thresh = 1.6 (default)
            #     if tlwh[2] * tlwh[3] > 100 and not vertical: #args.min_box_area = 100 (default)
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            #         results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n")
            #         print(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n") # frame id, tracking id, bounding box
            #         # cv2.putText(img, f"{class_name}: {score:.2f}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX,0.3, (36, 180, 120), 1)
            #                 # online_im = plot_tracking(
            #                 #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id,
            #                 #     fps=1. / timer.average_time
            #                 # )
            #
        # Display the image with bounding boxes
        # cv2.imshow(image_name, img)
        cv2.imwrite(out_file, img)  # Save the image to a file
        cv2.waitKey(0)  # Wait until a key is pressed to close the window
        cv2.destroyAllWindows()  # Close the window after key press



if __name__ == "__main__":
    main()

