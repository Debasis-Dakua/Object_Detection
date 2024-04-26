import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))

#input configuration for the model
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#read image
img = cv2.imread('img.jpg')

#detect obj in img
ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5)
print(ClassIndex)

#font style
font_scale =1
font = cv2.FONT_HERSHEY_PLAIN

#iterate over detected obj
for i in range(len(ClassIndex)):
    box = bbox[i]
    class_index = int(ClassIndex[i])
    conf = confidence[i]

    # return coordinates of bounding box
    x, y, w, h = box[0], box[1], box[2], box[3]

    #draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)

    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(classLabels[class_index - 1], font, fontScale=font_scale,thickness=1)

    # Calculate text position
    text_x = x
    text_y = y - 10 if y - 10 > text_height else y + 20

    # Draw text
    cv2.putText(img, classLabels[class_index - 1], (text_x, text_y), font, fontScale=font_scale, color=(0, 255, 0), thickness=1)



# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#showing the img plot
plt.imshow(img_rgb)
plt.axis('off')
plt.show()