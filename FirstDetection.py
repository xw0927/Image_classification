#https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
from imageai.Detection import ObjectDetection
import os
import pandas as pd

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


df = pd.read_csv('balanced_cars.csv')
with open(os.path.join(execution_path ,'object_detection2.csv'), 'w') as f:
    f.write("car_url_id,object_name,probability\n")
    for idx in range(711,len(df)): 
        car_url_id = df.loc[idx, 'car_url_id']
        image_file_name = '{}.png'.format(car_url_id)
        saved_image_file_name = '{}.png'.format(car_url_id)
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path ,'img',image_file_name), output_image_path=os.path.join(execution_path,'img_obj',saved_image_file_name ))
        for eachObject in detections:
            prob = float("{0:.2f}".format(eachObject["percentage_probability"]))
            print(eachObject["name"],eachObject["percentage_probability"])
            f.write("{},{},{}\n".format(car_url_id,eachObject["name"], prob))

