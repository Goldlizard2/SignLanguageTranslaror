import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np

# Define constants and paths
WORKSPACE_PATH = './Tensorflow/workspace'
SCRIPTS_PATH = './Tensorflow/scripts'
APIMODEL_PATH = './Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

# Function for detection
@tf.function
def detect_fn(image, model):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

# Function to visualize detections
def visualize_detections(image_np, detections, category_index):
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=0.8,
        agnostic_mode=False
    )
    return image_np_with_detections

# Main function to run the detection loop
def run_detection():
    category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor, detection_model)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = visualize_detections(image_np, detections, category_index)
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (640, 480)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    run_detection()

main()