from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '.././configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '.././faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

import cv2

labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }

import numpy as np

def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):
  draw_img = img_array.copy()
  bbox_color=(0, 255, 0)
  text_color=(0, 0, 0)
  dog_true = 0
  count = 0
  results = inference_detector(model, img_array)
  for result_ind, result in enumerate(results):
    if len(result) == 0:
      continue
    
    result_filtered = result[np.where(result[:, 4] > score_threshold)]
    
    for i in range(len(result_filtered)):
      left = int(result_filtered[i, 0])
      top = int(result_filtered[i, 1])
      right = int(result_filtered[i, 2])
      bottom = int(result_filtered[i, 3])
      if is_print:
        if labels_to_names_seq[result_ind] == "dog" and count == 0:
          caption = "{}: {:.2f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
          cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=10)
          cv2.putText(draw_img, caption, (int(left + 10), int(top + 60)), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 5)
          dog_true = 1
          count = 1

  return draw_img , dog_true

import matplotlib.pyplot as plt

img_arr = cv2.imread('./demo.jpg')
detected_img , dog_result = get_detected_img(model, img_arr,  score_threshold=0.3, is_print=True)
    
if dog_result==1:
    print("dog")
else:
    print("not_dog")

detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(detected_img)
plt.savefig('result.jpg')