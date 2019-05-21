import matplotlib as mpl
mpl.use('Qt5Agg')
import cv2
import json
# 展示COCO数据集的B_BOX，需要事先调用form_index_b-box.py形成index
index_file_path = 'D:/DataBackup/COCO/instances_train2014_index.json'
brief_file_path = 'D:/DataBackup/COCO/instances_train2014_brief.json'

with open(index_file_path, 'r') as f:
    index = json.load(f)
with open(brief_file_path, 'r') as f:
    brief = json.load(f)
    id_name_map = brief['id2name']

count = 0
for img_info in index:
    img_path = img_info[0]
    img_info.pop(0)
    img = cv2.imread(img_path)
    for bbox in img_info:
        [(x, y, w, h), cat] = bbox
        (x1, y1) = (int(x), int(y))
        (x2, y2) = (int(x + w), int(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), thickness=2)
        cv2.putText(img, id_name_map[str(cat)], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 200), 2)
    print(count)
    count += 1
    cv2.imshow('sdf', img)
    cv2.waitKey(1000)
