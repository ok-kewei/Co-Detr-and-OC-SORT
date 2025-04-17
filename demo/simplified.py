# Copyright (c) OpenMMLab. All rights reserved.
import os
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2
from projects import *
import csv


def main():
    config_file = 'projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
    checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Path to the folder containing images
    folder_path = '/home/kewei/rain_data/singapore/one/left_un_dist/'
    # folder_path = '/home/kewei/Co-DETR/images/'

    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
    # for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            out_file = os.path.join(os.path.dirname(img_path), 'codetr_out', image_name + ".png")
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

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
                if class_name in {'car', 'bus', 'truck'}:
                # Iterate over each detected bounding box for the class
                    for bbox in bboxes:
                        x1, y1, x2, y2, score = bbox
                        if score >= threshold:  # Apply score threshold
                            print(f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}, img:{image_name} ")
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 100, 0), thickness=2)
                            # cv2.putText(img, 'rer',(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            cv2.putText(img,  f"{class_name}: {score:.2f}", (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 180, 12), 1)
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

# import os
# from mmdet.apis import inference_detector, init_detector
# import cv2
#
# def draw_and_show_bounding_boxes(image_path, bboxes, class_name):
#     # Read the image with OpenCV
#     image = cv2.imread(image_path)
#
#     # Loop through bounding boxes
#     for bbox in bboxes:
#         x1, y1, x2, y2, score = bbox
#         if score >= 0.6 and class_name in {'car', 'bus', 'truck'}:
#             # Draw rectangle around detected object
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
#
#             # Create label with class name and confidence score
#             label = f"{class_name}: {score:.2f}"
#
#             # Calculate text width and height for background
#             (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#
#             # Draw a filled rectangle behind text for readability
#             cv2.rectangle(image, (int(x1), int(y1) - text_height - baseline),
#                           (int(x1) + text_width, int(y1)), (0, 255, 0), cv2.FILLED)
#
#             # Put the label text above the bounding box
#             cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             print(f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}")
#
#     # Display the image with bounding boxes
#     cv2.imshow('Image with Bounding Boxes', image)
#     cv2.waitKey(0)  # Wait until a key is pressed to close the window
#     cv2.destroyAllWindows()  # Close the window after key press
#
#
# def main():
#     # Specify the paths to config and checkpoint files
#     config_file = 'projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py'
#     checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'
#
#     # Initialize the model
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
#     # Path to the folder containing input images
#     # input_folder = '/home/kewei/rain_data/singapore/one/left_un_dist/'
#     input_folder = '/home/kewei/Co-DETR/images/'
#
#     # Loop through each image in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(input_folder, filename)
#             image_name = os.path.splitext(filename)[0]
#
#             # Perform inference on the image
#             result = inference_detector(model, image_path)
#             bbox_result, _ = result, None  # result[0] contains bbox predictions for each class
#
#             # Process and draw bounding boxes for each detected class
#             for class_id, bboxes in enumerate(bbox_result):
#                 # Skip if no detections for this class
#                 if bboxes.shape[0] == 0:
#                     continue
#
#                 # Get the class name
#                 class_name = model.CLASSES[class_id]
#
#                 # Draw bounding boxes and show image
#                 # draw_and_show_bounding_boxes(image_path, bboxes, class_name)
#                 image = cv2.imread(image_path)
#
#                 # Loop through bounding boxes
#                 for bbox in bboxes:
#                     x1, y1, x2, y2, score = bbox
#                     if score >= 0.6 and class_name in {'car', 'bus', 'truck'}:
#                         # Draw rectangle around detected object
#                         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
#
#                         # Create label with class name and confidence score
#                         label = f"{class_name}: {score:.2f}"
#
#                         # Calculate text width and height for background
#                         (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#
#                         # Draw a filled rectangle behind text for readability
#                         cv2.rectangle(image, (int(x1), int(y1) - text_height - baseline),
#                                       (int(x1) + text_width, int(y1)), (0, 255, 0), cv2.FILLED)
#
#                         # Put the label text above the bounding box
#                         cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#                         print(f"Class: {class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {score}")
#
#                 # Display the image with bounding boxes
#                 cv2.imshow('Image with Bounding Boxes', image)
#                 cv2.waitKey(0)  # Wait until a key is pressed to close the window
#                 cv2.destroyAllWindows()  # Close the window after key press
#
#
# if __name__ == "__main__":
#     main()


