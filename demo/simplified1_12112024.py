# Copyright (c) OpenMMLab. All rights reserved.
import os
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2
from projects import *
import csv
from ocsort import OCSort
from trackers.ocsort_tracker.ocsort import OCSort
# from trackers.tracking_utils.timer import Timer


def main():
    config_file = 'projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
    checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    oc_sort = OCSort(det_thresh=0.6)

    # Path to the folder containing images
    folder_path = '/home/kewei/rain_data/singapore/one/left_un_dist/'
    # folder_path = '/home/kewei/Co-DETR/images/oxford'
    frame_id = 0
    results = []
    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
    # for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            out_file = os.path.join(os.path.dirname(img_path), 'codetr_out', image_name + ".png")
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            frame_id = frame_id +1
            # Run inference
            result = inference_detector(model, img_path)
            # Extract and print class names, bounding boxes, and scores
            bbox_result, _ = result, None  # result[0] contains bbox predictions for each class

            # Iterate over all classes
            for class_id, bboxes in enumerate(bbox_result):
                # Skip if there are no detections for the class
                if bboxes.shape[0] == 0:
                    continue

                # Filter out the confidence level
                threshold = 0.5
                bboxes = bboxes[bboxes[:, 4] > threshold]
                # Get class name
                class_name = model.CLASSES[class_id]
                if class_name in {'car', 'bus', 'truck','person'}:
                # Iterate over each detected bounding box for the class
                    for bbox in bboxes:
                        x1, y1, x2, y2, score = bbox
                        if score >= threshold:  # Apply score threshold
                            print(f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img:{image_name} , frame id: {frame_id}")
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 100, 0), thickness=1)
                            # # cv2.putText(img, 'rer',(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            cv2.putText(img,  f"{class_name}: {score:.2f}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 180, 120), 1)


                            bbox = bbox.reshape((1,5))
                            # bbox = bbox[:, :-1] # remove the last element, confidence level
                            # bbox = bbox.reshape((1,4))

            # Here, you can pass the `bboxes` to OC-SORT for tracking
            tracks = oc_sort.update(bbox, [height, width], [height, width])
            # tracks = oc_sort.update(bbox, class_name, frame_id)
            online_tlwhs = []
            online_ids = []
            print("tracks:", tracks)
            for t in tracks:
                print("t", t)
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6 #args.aspect_ratio_thresh = 1.6 (default)
                if tlwh[2] * tlwh[3] > 100 and not vertical: #args.min_box_area = 100 (default)
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n")
                    print(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n") # frame id, tracking id, bounding box
                    # cv2.putText(img, f"{class_name}: {score:.2f}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX,0.3, (36, 180, 120), 1)
                            # online_im = plot_tracking(
                            #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id,
                            #     fps=1. / timer.average_time
                            # )

            # Display the image with bounding boxes
            cv2.imshow(image_name, img)
            cv2.imwrite(out_file, img)  # Save the image to a file
            cv2.waitKey(0)  # Wait until a key is pressed to close the window
            cv2.destroyAllWindows()  # Close the window after key press
                #         #   # # Show and save the filtered results
                        #
                        # show_result_pyplot(
                        #     model,
                        #     img,
                        #     bbox,
                        #     palette='coco',
                        #     score_thr=0.6,
                        #     out_file=out_file)



if __name__ == "__main__":
    main()

# # Copyright (c) OpenMMLab. All rights reserved.
# import os
#
# from mmdet.apis import (async_inference_detector, inference_detector,
#                         init_detector, show_result_pyplot)
#
# import cv2
#
# from ocsort import OCSort
# from image_demo import *
#
#
# def inference(self, img, timer):
#     img_info = {"id": 0}
#     if isinstance(img, str):
#         img_info["file_name"] = osp.basename(img)
#         img = cv2.imread(img)
#     else:
#         img_info["file_name"] = None
#
#     height, width = img.shape[:2]
#     img_info["height"] = height
#     img_info["width"] = width
#     img_info["raw_img"] = img
#
#     img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
#     img_info["ratio"] = ratio
#     img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
#     if self.fp16:
#         img = img.half()  # to FP16
#
#     with torch.no_grad():
#         timer.tic()
#         outputs = self.model(img)
#         if self.decoder is not None:
#             outputs = self.decoder(outputs, dtype=outputs.type())
#         outputs = postprocess(
#             outputs, self.num_classes, self.confthre, self.nmsthre
#         )
#     return outputs, img_info
#
# def main():
#     oc_sort = OCSort(det_thresh=0.6 )
#     # timer = Timer()
#     config_file = 'projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
#     checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
#     # build the model from a config file and a checkpoint file
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
#     # Define vehicle classes of interest
#     vehicle_classes = ['car', 'truck', 'bus']
#
#     # Path to the folder containing images
#     folder_path = '/home/kewei/rain_data/singapore/one/left_un_dist/'
#
#     # Loop through all files in the folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(folder_path, filename)
#             image_name = os.path.splitext(os.path.basename(img_path))[0]
#             img = cv2.imread(img_path)
#             height, width = img.shape[:2]
#             # img_info["height"] = height
#             # img_info["width"] = width
#             out_file = os.path.join(os.path.dirname(img_path), 'codetr_out', image_name + ".png")
#             # Run inference
#             result = inference_detector(model, img_path)
#             # Extract and print class names, bounding boxes, and scores
#             bbox_result, _ = result, None  # result[0] contains bbox predictions for each class
#
#             # Iterate over classes and filter based on vehicle_classes and score
#             for class_id, bboxes in enumerate(bbox_result):
#                 if bboxes.shape[0] == 0:
#                     continue
#
#                 class_name = model.CLASSES[class_id]
#             # Iterate over all classes
#             for class_id, bboxes in enumerate(bbox_result):
#                 # Skip if there are no detections for the class
#                 if bboxes.shape[0] == 0:
#                     continue
#
#                 # Get class name
#                 class_name = model.CLASSES[class_id]
#
#                 # Iterate over each detected bounding box for the class
#                 for bbox in bboxes:
#                     x1, y1, x2, y2, score = bbox
#                     if score >= 0.6 and class_name in {'car', 'bus','truck'}:  # Apply score threshold
#                         print(f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img:{image_name} ")
#
#                         # Here, you can pass the `bboxes` to OC-SORT for tracking
#                         tracks = oc_sort.update(bbox, [height, width], [height, width])
#
#                         # Draw tracked boxes on the frame
#                         for track in tracks:
#                             # Using cv2.rectangle() method
#                             # Draw a rectangle with blue line borders of thickness of 2 px
#                             # image = cv2.rectangle(image, start_point, end_point, color, thickness)
#                             cv2.rectangle(img, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])),
#                                           (0, 255, 0), 2)
#
#                         cv2.imshow('Frame', img)
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
#
#                         #   # # Show and save the filtered results
#                         #
#                         # show_result_pyplot(
#                         #     model,
#                         #     img,
#                         #     bbox,
#                         #     palette='coco',
#                         #     score_thr=0.6,
#                         #     out_file=out_file)
#
#             # # Prepare filtered results structure for specified vehicle classes
#             # filtered_result = [None] * len(model.CLASSES)
#             # for class_id, bboxes in enumerate(bbox_result):
#             #     # Only consider specified vehicle classes
#             #     if model.CLASSES[class_id] in vehicle_classes:
#             #         # Filter bboxes by score
#             #         filtered_bboxes = [bbox for bbox in bboxes if bbox[4] >= 0.6]
#             #         if filtered_bboxes:
#             #             filtered_result[class_id] = filtered_bboxes
#             #             # Print each filtered detection for verification
#             #             for bbox in filtered_bboxes:
#             #                 x1, y1, x2, y2, score = bbox
#             #                 print(f"Class: {model.CLASSES[class_id]}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img: {image_name}")
#             #
#             # # Plot and save only if there are any filtered detections
#             # if any(filtered_result):
#             #     show_result_pyplot(
#             #         model,
#             #         img,
#             #         result,
#             #         score_thr=0.6,
#             #         out_file=out_file
#             #     )
#
#
# if __name__ == "__main__":
#     main()


# import cv2
#
# from ocsort import OCSort
# from image_demo import *
#
# def main():
#     oc_sort = OCSort()
#
#     cap = cv2.VideoCapture('input_video.mp4')
#
#     ret, frame = cap.read()
#     prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         detections = detect_objects(frame)  # Use any object detection model
#
#         if len(detections) > 0:
#             tracks = oc_sort.update(detections)
#
#         # Optical flow tracking
#         prev_points = np.array([[d[0], d[1]] for d in tracks], dtype=np.float32)
#         prev_points, curr_points = calculate_optical_flow(prev_frame_gray, curr_frame_gray, prev_points)
#
#         prev_frame_gray = curr_frame_gray
#
#         for point in curr_points:
#             cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
#
#         cv2.imshow('Tracking', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()