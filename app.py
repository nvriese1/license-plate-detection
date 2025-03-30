import os
import json
import gradio as gr
import onnxruntime as ort

from data_objects import (
    YOLOXDetector,
    Detection,
    ObjectDetectionConfig,
)

# Model configs
object_detection_config = ObjectDetectionConfig()

# Load object detector
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
object_detector = YOLOXDetector(
	model_path=object_detection_config.object_detection_model_path,
	input_shape=object_detection_config.input_shape,
	confidence_threshold=object_detection_config.confidence_threshold,
	providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
	sess_options=sess_options,
)

def generate_json(detected_objects):
    detections_list = []
    for obj in detected_objects:
        detections_list.append({
            "class_name": obj.display_name,
            "score": obj.score,
            "bbox_xyxy": obj.points_xyxy.tolist()
        })

    json_data = json.dumps(detections_list, indent=4)
    with open("detections.json", "w") as f:
        f.write(json_data)

    return "detections.json", json_data

def predict(input_img):

    final_boxes, final_scores, final_cls = object_detector.predict(input_img)

    detected_objects = [
        Detection(
            points=bbox,
            score=score,
            class_id=class_id,
            color=object_detection_config.color_map.get(class_id),
            display_name=object_detection_config.display_map.get(class_id),
            centroid_thickness=-1,
            centroid_radius=5
        )
        for class_id in list(object_detection_config.class_map.keys())
        for bbox, score in zip(final_boxes[final_cls == class_id], final_scores[final_cls == class_id])
    ]

    for obj in detected_objects:
        input_img = obj.draw(
            image=input_img,
            draw_boxes=True,
            draw_centroids=False,
            draw_text=True,
            draw_projections=False,
            box_display_type="minimal",
            fill_text_background=True,
            box_line_thickness=2,
            box_corner_length=15,
            text_scale=0.6,
            obfuscate_classes=[],
        )
    
    json_file, json_text = generate_json(detected_objects)

    return input_img, {obj.display_name: obj.score for obj in detected_objects}, json_file, json_text

example_images = [
    os.path.join("./examples", img) for img in os.listdir("./examples") if img.lower().endswith(('png', 'jpg', 'jpeg'))
]

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select image to process", sources=['upload', 'webcam'], type="numpy"),
    outputs=[
        gr.Image(label="Processed Image"), 
        gr.Label(label="Result", num_top_classes=2),
        gr.File(label="Download JSON"),
        gr.Textbox(label="Copy JSON Text", lines=10)
    ],
    title="License Plate Detection",
    examples=example_images,
)

if __name__ == "__main__":
    gradio_app.launch(share=True)