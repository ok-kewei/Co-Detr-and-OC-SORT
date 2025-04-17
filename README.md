## OC-SORT + Co-DETR Integration

An integration of [OC-SORT tracking](https://github.com/noahcao/OC_SORT) with [Co-DETR detection](https://github.com/Sense-X/Co-DETR).

---

### 🔗 Pretrained Model

To get the pretrained model, please refer to the [Co-DETR GitHub main page](https://github.com/Sense-X/Co-DETR).

In this example, I used:

checkpoint_file = 'model/co_dino_5scale_lsj_swin_large_3x_coco.pth'

### ▶️ Running the Code

To run the demo, check:

```bash
demo/simplified1.py
```


Customizations that I did:
1. Class filter: Only tracks {'car', 'bus', 'truck'} (see line 42)
2. Detection threshold: 0.3 (see line 38)
3. Tracking threshold: 0.4 (see line 74)

### Result: 
The tracking id is able to restore after visual disruption of a sequence of 5 image frames (of no tracking id assigned) with the setting of Min Age = 30.  
![image](https://github.com/user-attachments/assets/dd0aa2c7-7c57-4f12-aa2d-263b152a2d37)
