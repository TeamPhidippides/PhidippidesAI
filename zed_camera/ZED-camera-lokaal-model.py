#!/usr/bin/env python
# coding: utf-8
import math as mt
import time
import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import os
import tarfile
import urllib.request
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import pyzed.sl as sl


def getDepthFromBox(coordinates, depthMap): 
    minXCoordinate = depthMap.get_width() * coordinates[1]
    minYCoordinate = depthMap.get_height() * coordinates[0]
    maxXCoordinate = depthMap.get_width() * coordinates[3]
    maxYCoordinate = depthMap.get_height() * coordinates[2]
    
    xCoordinate = int(round((minXCoordinate + maxXCoordinate) / 2))
    yCoordinate = int(round((minYCoordinate + maxYCoordinate) / 2))
    err, distance = depthMap.get_value(yCoordinate, xCoordinate)
    return distance



def load_image_into_numpy_array(image): #maakt van een Mat.sl() een Numpy array
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


@tf.function
def detect_fn(image):   #Functie voor de object detectie
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def main():
    init = sl.InitParameters()  
    init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depthMap = sl.Mat()
    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT) #vraag een frame op van de camera
            cam.retrieve_measure(depthMap, sl.MEASURE.DEPTH)    #vraag de dieptekaart op van de camera
            image_np = load_image_into_numpy_array(image)       #maak van het plaatje een numpy array

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) #maakt een tensor aan voor de object detectie
            detections, predictions_dict, shapes = detect_fn(input_tensor)  #start de object detectie

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(    #visualiseert de object detectie
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=15.0,
                min_score_thresh=.30,
                agnostic_mode=False)
            
            boxes = detections['detection_boxes'][0].numpy()    #print de afstand tot alle objecten
            for i in range (len(detections['detection_boxes'][0].numpy())):
                if (detections['detection_scores'][0].numpy()[i]) != 0:
                    print("Object:", i + 1, "distance", getDepthFromBox(detections['detection_boxes'][0].numpy()[i], depthMap))
            print("======================================")
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
            cv2.waitKey(5)
        else:
            key = cv.waitKey(5)
    cv.destroyAllWindows()

DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)


#laad het model
MODEL_NAME = 'V5'
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

#zorgt er voor dat het gpu geheugen aangepast wordt aan het device
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
if __name__ == '__main__':
    main()

