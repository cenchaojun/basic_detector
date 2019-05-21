import os
import matplotlib as mpl
mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json

data_root = 'D:/DataBackup/VOC2012'
data_folder = data_root + '/JPEGImages'
ann_folder = data_root + '/Annotations'

index_file = data_root + '/VOC2012_index.json'
data_brief_file = data_root + '/VOC2012_brief.json'

file_list = os.listdir(data_folder)
name_list = ['person',
        'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
name2id = {name: index for (index, name) in enumerate(name_list)}
id_name_map = {value: key for key, value in name2id.items()}

# 提取xml文件，获得bbox、标签
index_content = []
for img_file in file_list:
    content = {}
    (img_name, extension) = os.path.splitext(img_file)
    xml_file = ann_folder + '/' + img_name + '.xml'
    # 开始提取
    tree = et.parse(xml_file)
    root = tree.getroot()
    # 获得文件路径
    file_name = root.find('filename')
    file_path = data_folder + '/' + file_name.text
    content['file_path'] = file_path
    content['img_id'] = img_name
    content['dets'] = []
    objects = root.findall('object')
    # 获得bbox和label
    for obj in objects:
        bbox = obj.find('bndbox')
        try:
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            name = obj.find('name').text
            if name not in name2id.keys():
                raise Exception('Wrong Name: %s in file %s' % (name, file_path))
            content['dets'].append([[xmin, ymin, xmax, ymax], name2id[name]])
        except ValueError:
            print(('Wrong Value in file %s' % (file_path)))
    index_content.append(content)

with open(index_file, 'wt+') as f:
    json.dump(index_content, f)
    print('save index %s' % index_file)

# 记录xml

with open(data_brief_file, 'wt+') as f:
    brief = {}
    brief['data_path'] = data_folder
    brief['ann_file'] = ann_folder
    brief['index_file'] = index_file
    brief['total number'] = len(index_content)
    brief['format of index file'] = \
        '[{\'file_path\' : ..., \', img_id(str)\': ..., ' \
        '\'dets\': [[b-box(x1, x2, y1, y2 4 int)], label(integral)], ...], ...]'
    brief['id2name'] = id_name_map
    json.dump(brief, f)
    print('save brief intro %s' % data_brief_file)


a = 0