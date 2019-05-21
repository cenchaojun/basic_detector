import matplotlib as mpl
mpl.use('Qt5Agg')
import cv2
import json
# 展示COCO数据集的B_BOX，需要事先调用form_index_b-box.py形成index
index_file_path = 'D:/DataBackup/VOC2012/VOC2012_index.json'
brief_file_path = 'D:/DataBackup/VOC2012/VOC2012_brief.json'
priors_file_path = '../targets/priors.json'
target_root = '../targets'

with open(index_file_path, 'r') as f:
    index = json.load(f)
with open(brief_file_path, 'r') as f:
    brief = json.load(f)
    id_name_map = brief['id2name']

count = 0
with open(priors_file_path, 'r') as f:
    priors = json.load(f)
    print('load priors done')

for img_info in index:
    img_path = img_info['file_path']
    img = cv2.imread(img_path)
    img_id = img_info['img_id']
    target_idx_file = target_root + '/' + img_id + '.json'
    with open(target_idx_file, 'r') as f:
        target_idx = json.load(f)
    for (i, cat) in enumerate(target_idx):
        if cat == 0:
            # print('pass %d' % i)
            continue
        [x1, y1, x2, y2] = priors[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), thickness=2)
        cv2.putText(img, id_name_map[str(cat-1)], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 200), 2)
    print(count)
    print(img_id)
    count += 1
    cv2.imshow('sdf', img)
    cv2.waitKey(1000)
