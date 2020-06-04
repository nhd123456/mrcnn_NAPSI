import cv2
import numpy as np

def random_colors(N):
    np.random.seed(1)
    colors = [tuple( 255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply mask"""
    
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):

    n_instances = boxes.shape[0]
    
    if not n_instances:
        print('No instances to display')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        
    colors = random_colors(n_instances)
    height, width = image.shape[:2]
    
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
            
        y1, x1, y2, x2 = boxes[i] #box  左上右下座標
        Xc = (x1 + x2) // 2
        Yc = (y1 + y2) // 2
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        image = cv2.line(image, (Xc, y1), (Xc, y2), (255, 255, 255), 2)
        image = cv2.line(image, (x1, Yc), (x2, Yc), (255, 255, 255), 2)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.3f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
        
    return image

if __name__ == '__main__':
    import os
    import sys
    import random
    import math
    import time
    #import coco
    #import utils
    import skimage.io
    import model as modellib
    
    ROOT_DIR = os.path.abspath("/")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    
    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    results = model.detect([image], verbose=1)

    r = results[0]
    output = display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imshow('output', output)