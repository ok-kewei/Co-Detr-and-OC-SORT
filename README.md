## OC-SORT + Co-DETR Integration

This work combines [OC-SORT tracking](https://github.com/noahcao/OC_SORT) for object tracking with [Co-DETR detection](https://github.com/Sense-X/Co-DETR) as the end-to-end object detector.

---

### 🔗 Pretrained Model

To get the pretrained model, please refer to the [Co-DETR GitHub main page](https://github.com/Sense-X/Co-DETR). Download the model (.pth) file and save it under model folder. 

In this example, I used checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth' as it (Co-DINO 	backbone:Swin-L 	36 	LSJ 	COCO 	60.7) scores the highest in coco dataset of 60.7 box AP. 

### Running the Code

To run the demo, check:

```bash
demo/simplified1.py
```

Changes made in this demo/simplified1 file:
1. Class filter: Only tracks {'car', 'bus', 'truck'} (see line 42)
2. Detection threshold: 0.3 (see line 38)
3. Tracking threshold: 0.4 (see line 74)

### Result: 
The tracking id is able to restore after visual disruption of a sequence of 5 image frames (of no tracking id assigned) with the setting of Min Age = 30.  
![image](https://github.com/user-attachments/assets/dd0aa2c7-7c57-4f12-aa2d-263b152a2d37)
