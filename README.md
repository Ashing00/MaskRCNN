# MaskRCNN
provide example code how to train Maskrcnn from https://github.com/matterport/Mask_RCNN



1.prepare folder <pic> <json> in <train_data_mold> folder
    -<train_data_mold> is your training data folder with <pic> <json>
    -<pic> is your training data picture
    -<json> is your training data .json file from "labelme" tool generated
    
2.use  json_to_dataset.py  create  <cv2_mask> <labelme_json>
    -python json_to_dataset_ok.py train_data_mold/json
    -it will create <cv2_mask> <labelme_json> in train_data_mold/json
    -copy <cv2_mask> <labelme_json> in  <train_data_mold>
    
3. motify train_mold.py and run  to train model
  -you can comapre train_mold.py and train_bread.py to look what is the difference between them
   and know how to modify the parameter
   
4.run test_mold.py
--you can comapre comapre test_mold.py and test_bread.py to look what is the difference between them

5.training weights will be generated in <logs> folder
