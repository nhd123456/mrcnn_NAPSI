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
        label = names[ids[i]]
        
        if (label == 'nail'):
            image = cv2.line(image, (Xc, y1), (Xc, y2), (0, 0, 0), 2)
            image = cv2.line(image, (x1, Yc), (x2, Yc), (0, 0, 0), 2)
            text = 'NAPSI Score \n Nail Bed:2\n Nail Matrix:0\n Total:2'
            y0 = 15
            for u, line in enumerate(text.split('\n')):
                y = 20 + u * y0
                cv2.putText(image, line, (210, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            score = scores[i] if scores is not None else None
            caption = '{} {:.3f}'.format(label, score) if score else label
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2
            )
        
    return image

if __name__ == '__main__':
    import os
    import sys
    import random
    import math
    import time
    #import utils
    import skimage.io
    from mrcnn import model as modellib
    import napsi

    ROOT_DIR = os.path.abspath("")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    IMAGE_DIR = os.path.join(ROOT_DIR, "dataset/test3")
    napsi_MODEL_PATH = os.path.join(ROOT_DIR, "logs/napsi20190705T1623/mask_rcnn_napsi_0050.h5")
    #sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  
   
    
    class InferenceConfig(napsi.napsiConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(napsi_MODEL_PATH, by_name=True)
    
    class_names = ['BG', 'nail']
    
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    results = model.detect([image], verbose=0)
    
    
    r = results[0]
    output = display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows
