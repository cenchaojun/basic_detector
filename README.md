# basic_detector
detector for experiment

# 1.文件说明
1. COCO_tool:
	1. evalDemo.py: 评估demo
	2. form_index_b-box.py: 形成index
	3. show_b-box.py: 展示bbox
	4. ...fakebbox100... : evalDemo中使用的fake结果
2. PASCLA_tool
	1. form_index_b-box.py： 同上
	2. show_b-box.py: 同上
	3. VOC_eval.py：评估demo
3. ssd.pytorch：来源于github上的项目，用于参考
4. ssd_tool:
	1. config.py
	2. match.py : match function
	3. prior_box ： generate prior box(anchor)
	4. target_box.py: generate target boxes and save them
	5. target_box-not lable.py:just for test, useless
	6. show-target_box.py: 展示target box
5. train.py : train
6. test_tool.py : some tools for test bbox
7. ...
