import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Union, Literal, Dict
from pydantic import BaseModel

# Configuration for YOLOX model, set path to model / class - name mappings here!
class ObjectDetectionConfig(BaseModel):
	"""Configuration for trained YOLOX object detection model."""
	
	# Model path & hyperparameters
	object_detection_model_path: str = "./models/yolox_custom-plates-2cls-0.1.onnx"
	confidence_threshold: float = 0.50
	nms_threshold: float = 0.65
	input_shape: Tuple[int] = (640, 640)
	
	# Class specific inputs
	class_map: Dict = {0: 'license-plates', 1: 'License_Plate'}
	display_map: Dict = {0: 'license-plate', 1: 'license-plate'}
	color_map: Dict = {0: (186, 223, 255), 1: (100, 255, 255)}

class Detection:
	def __init__(
		self,
		points: np.ndarray,
		class_id: Union[int, None] = None,
		score: Union[float, None] = 0.0,
		color: Tuple[int, int, int] = (100, 255, 255),
		display_name: str = "Box",
		centroid_radius: int = 5,
		centroid_thickness: int = -1
	):
		"""
		Represents an object detection in the scene.
		Stores bounding box, class_id, and other attributes for tracking and visualization.
		"""
		self.points_xyxy = points
		self.class_id = class_id
		self.score = score
		self.color_bbox = color
		self.color_centroid = color
		self.radius_centroid = centroid_radius
		self.thickness_centroid = centroid_thickness
		self.centroid_location: str = "center"
		self.display_name: str = display_name
		self.track_id: int = None
		self.id: int = None
		self.active: bool = False
		self.status: str = ""

	def __repr__(self) -> str:
		return f"Detection({str(self.display_name)})"

	@property
	def bbox_xyxy(self) -> np.ndarray:
		return self.points_xyxy

	@property
	def size(self) -> float:
		"""Return the bounding box area in pixels."""
		x1, y1, x2, y2 = self.points_xyxy
		return (x2 - x1) * (y2 - y1)

	def bbox_image(self, image: np.ndarray, buffer: int = 0) -> np.ndarray:
		"""Extract the image patch corresponding to this detection"s bounding box."""
		x1, y1, x2, y2 = self.points_xyxy
		height, width = image.shape[:2]
		x1 = max(0, int(x1 - buffer))
		y1 = max(0, int(y1 - buffer))
		x2 = min(width, int(x2 + buffer))
		y2 = min(height, int(y2 + buffer))
		return image[y1:y2, x1:x2]

	def centroid(self, location: str = None) -> np.ndarray:
		"""Get the centroid of the bounding box based on the chosen centroid location."""
		if location is None:
			location = self.centroid_location
		x1, y1, x2, y2 = self.points_xyxy
		if location == "center":
			centroid_loc = [(x1 + x2) / 2, (y1 + y2) / 2]
		elif location == "top":
			centroid_loc = [(x1 + x2) / 2, y1]
		elif location == "bottom":
			centroid_loc = [(x1 + x2) / 2, y2]
		elif location == "left":
			centroid_loc = [x1, (y1 + y2) / 2]
		elif location == "right":
			centroid_loc = [x2, (y1 + y2) / 2]
		elif location == "upper-left":
			centroid_loc = [x1, y1]
		elif location == "upper-right":
			centroid_loc = [x2, y1]
		elif location == "bottom-left":
			centroid_loc = [x1, y2]
		elif location == "bottom-right":
			centroid_loc = [x2, y2]
		else:
			raise ValueError("Unsupported location type.")
		return np.array([centroid_loc], dtype=np.float32)

	def draw(
		self,
		image: np.ndarray,
		draw_boxes: bool = True,
		draw_centroids: bool = True,
		draw_text: bool = True,
		draw_projections: bool = False,
		fill_text_background: bool = False,
		box_display_type: Literal["minimal", "standard"] = "standard",
		box_line_thickness: int = 2,
		box_corner_length: int = 20,
		obfuscate_classes: List[int] = [],
		centroid_color: Union[Tuple[int, int, int], None] = None,
		centroid_radius: Union[int, None] = None,
		centroid_thickness: Union[int, None] = None,
		text_position_xy: Tuple[int] = (25, 25),
		text_scale: float = 0.8,
		text_thickness: int = 2,
	) -> np.ndarray:
		"""Draw bounding boxes and centroids for the detection.

		If fill_text_background is True, the text placed near the centroid is drawn over a blurred 
		background extracted from the image. Extra padding is added so the background box is taller.
		"""
		image_processed = image.copy()

		if draw_boxes:
			object_bbox: np.ndarray = self.bbox_xyxy
			bbox_color: Tuple[int, int, int] = self.color_bbox if self.color_bbox is not None else (100, 255, 255)
			if object_bbox is not None:
				
				x0 = int(object_bbox[0])
				y0 = int(object_bbox[1])
				x1 = int(object_bbox[2])
				y1 = int(object_bbox[3])

				if self.class_id in obfuscate_classes:
					roi = image_processed[y0:y1, x0:x1]
					if roi.size > 0:
						image_processed[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (61, 61), 0)
	
				if box_display_type.strip().lower() == "minimal":
					box_corner_length = int(
						min(box_corner_length, (x1 - x0) / 2, (y1 - y0) / 2)
					)
					cv2.line(image_processed, (x0, y0), (x0 + box_corner_length, y0), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x0, y0), (x0, y0 + box_corner_length), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x1, y0), (x1 - box_corner_length, y0), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x1, y0), (x1, y0 + box_corner_length), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x0, y1), (x0 + box_corner_length, y1), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x0, y1), (x0, y1 - box_corner_length), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x1, y1), (x1 - box_corner_length, y1), color=bbox_color, thickness=box_line_thickness)
					cv2.line(image_processed, (x1, y1), (x1, y1 - box_corner_length), color=bbox_color, thickness=box_line_thickness)
	
				elif box_display_type.strip().lower() == "standard":
					cv2.rectangle(
						image_processed,
						(x0, y0),
						(x1, y1),
						color=bbox_color,
						thickness=box_line_thickness
					)

		if draw_projections:
	
			projection_start_centroid: np.ndarray = self.centroid(location="bottom")[0]
			if self.velocity is not None:
				projection_end_centroid: np.array = np.array([self.centroid(location="bottom")[0] + self.velocity])[0]
			else:
				projection_end_centroid = projection_start_centroid
			projection_start_coords: Tuple[int, int] = (int(projection_start_centroid[0]), int(projection_start_centroid[1]))
			projection_end_coords: Tuple[int, int] = (int(projection_end_centroid[0]), int(projection_end_centroid[1]))
		
			cv2.arrowedLine(
				image_processed, 
				projection_start_coords, 
				projection_end_coords, 
				color=(100, 255, 255),
				thickness=3,
				tipLength=0.2
			)

		centroid: np.ndarray = self.centroid()[0]
		centroid_coords: Tuple[int, int] = (int(centroid[0]), int(centroid[1]))
		if centroid_color is None:
			centroid_color = self.color_centroid
		if centroid_radius is None:
			centroid_radius = self.radius_centroid
		if centroid_thickness is None:
			centroid_thickness = self.thickness_centroid

		if draw_centroids:
	
			cv2.circle(
				image_processed,
				centroid_coords,
				centroid_radius,
				centroid_color,
				centroid_thickness,
				lineType=cv2.LINE_AA
			)

		if draw_text:
	
			display_text: str = str(self.display_name)
			text_position: Tuple[int, int] = (
				centroid_coords[0] + text_position_xy[0], 
				centroid_coords[1] + text_position_xy[1]
			)

			if hasattr(self, "score") and self.score:
				display_text += f" ({self.score})"
		
			if hasattr(self, "status") and self.status:
				display_text += f" ({self.status})"
				if self.status == "Waiting":
					display_text += f" ({int(self.queue_time_duration)}s)"
	
			if fill_text_background:
				font = cv2.FONT_HERSHEY_SIMPLEX
				(text_width, text_height), baseline = cv2.getTextSize(display_text, font, text_scale, text_thickness)
				pad_x = 0
				pad_y = 10
				# Calculate rectangle coordinates
				rect_x1 = text_position[0] - pad_x
				rect_y1 = text_position[1] - text_height - pad_y
				rect_x2 = text_position[0] + text_width + pad_x
				rect_y2 = text_position[1] + baseline + pad_y
				# Ensure coordinates are within image boundaries
				rect_x1 = max(0, rect_x1)
				rect_y1 = max(0, rect_y1)
				rect_x2 = min(image_processed.shape[1], rect_x2)
				rect_y2 = min(image_processed.shape[0], rect_y2)
				# Extract the region of interest and apply a Gaussian blur
				roi = image_processed[rect_y1:rect_y2, rect_x1:rect_x2]
				if roi.size > 0:
					image_processed[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.GaussianBlur(roi, (31, 31), 0)
	
			cv2.putText(
				image_processed,
				display_text,
				text_position,
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=text_scale,
				color=centroid_color,
				thickness=text_thickness,
				lineType=cv2.LINE_AA
			)

		return image_processed

class YOLOXDetector:
	def __init__(
		self,
		model_path: str,
		input_shape: Tuple[int] = (640, 640),
		confidence_threshold: float = 0.6,
		nms_threshold: float = 0.65,
		providers: List[str] = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
		sess_options=ort.SessionOptions(),
	):
		self.model_path: str = model_path
		self.dims: Tuple[int] = input_shape
		self.ratio: float = 1.0
		self.confidence_threshold: float = confidence_threshold
		self.nms_threshold: float = nms_threshold
		self.classes: List[str] = ["license-plates", "License_Plate"]
		self.categories: List[str] = ["DEFAULT" for _ in range(len(self.classes))]
		self.providers: List[str] = providers
		self.session = ort.InferenceSession(
				self.model_path,
				providers=self.providers,
				sess_options=sess_options,
		)

	def nms(self, boxes, scores, nms_thr):
		"""Single class NMS implemented in Numpy."""
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= nms_thr)[0]
			order = order[inds + 1]

		return keep

	def multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
		"""Multiclass NMS implemented in Numpy. Class-aware version."""
		final_dets = []
		num_classes = scores.shape[1]
		for cls_ind in range(num_classes):
			cls_scores = scores[:, cls_ind]
			valid_score_mask = cls_scores > score_thr
			if valid_score_mask.sum() == 0:
				continue
			else:
				valid_scores = cls_scores[valid_score_mask]
				valid_boxes = boxes[valid_score_mask]
				keep = self.nms(valid_boxes, valid_scores, nms_thr)
				if len(keep) > 0:
					cls_inds = np.ones((len(keep), 1)) * cls_ind
					dets = np.concatenate(
							[valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
					)
					final_dets.append(dets)
		if len(final_dets) == 0:
			return None
		return np.concatenate(final_dets, 0)


	def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
		"""Multiclass NMS implemented in Numpy. Class-agnostic version."""
		cls_inds = scores.argmax(1)
		cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

		valid_score_mask = cls_scores > score_thr
		if valid_score_mask.sum() == 0:
			return None
		valid_scores = cls_scores[valid_score_mask]
		valid_boxes = boxes[valid_score_mask]
		valid_cls_inds = cls_inds[valid_score_mask]
		keep = self.nms(valid_boxes, valid_scores, nms_thr)
		if keep:
			dets = np.concatenate(
					[valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
			)
		return dets

	def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=False):
		"""Multiclass NMS implemented in Numpy"""
		if class_agnostic:
			return self.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)
		else:
			return self.multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr)

	def preprocess(self, image: np.ndarray, bgr2rgb: bool = False):
		"""Preprocess image for YOLOX model."""
		if len(image.shape) == 3:
				padded_image = np.ones((self.dims[0], self.dims[1], 3), dtype=np.uint8) * 114
		else:
				padded_image = np.ones(self.dims, dtype=np.uint8) * 114

		if bgr2rgb:
				padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

		self.ratio = min(self.dims[0] / image.shape[0], self.dims[1] / image.shape[1])
		resized_image = cv2.resize(
				image,
				(int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio)),
				interpolation=cv2.INTER_LINEAR,
		).astype(np.uint8)
		padded_image[: int(image.shape[0] * self.ratio), : int(image.shape[1] * self.ratio)] = resized_image

		padded_image = padded_image.transpose((2, 0, 1))
		padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
		return padded_image

	def postprocess(self, outputs, p64=False):
		"""Post-process YOLOX model outputs into usable bounding boxes and scores."""
		grids = []
		expanded_strides = []
		strides = [8, 16, 32] if not p64 else [8, 16, 32, 64]

		hsizes = [self.dims[0] // stride for stride in strides]
		wsizes = [self.dims[1] // stride for stride in strides]

		for hsize, wsize, stride in zip(hsizes, wsizes, strides):
			xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
			grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
			grids.append(grid)
			shape = grid.shape[:2]
			expanded_strides.append(np.full((*shape, 1), stride))

		grids = np.concatenate(grids, 1)
		expanded_strides = np.concatenate(expanded_strides, 1)
		outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
		outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

		outputs = outputs[0]

		boxes = outputs[:, :4]
		scores = outputs[:, 4:5] * outputs[:, 5:]

		boxes_xyxy = np.ones_like(boxes)
		boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
		boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
		boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
		boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
		boxes_xyxy /= self.ratio
		return boxes_xyxy, scores

	def predict(self, image: np.ndarray):
		"""Run YOLOX detector on an image and return detected bounding boxes and scores."""
		image = self.preprocess(image=image)
		onnx_pred = self.session.run(None, {self.session.get_inputs()[0].name: np.expand_dims(image, axis=0)})[0]
		boxes_xyxy, scores = self.postprocess(onnx_pred)
		detections = self.multiclass_nms(
			boxes=boxes_xyxy,
			scores=scores,
			nms_thr=self.nms_threshold,
			score_thr=self.confidence_threshold,
			class_agnostic=False if len(self.classes) > 1 else True
		)
		if detections is not None and len(detections) > 0:
			final_boxes, final_scores, final_cls_inds = detections[:, :4], detections[:, 4], detections[:, 5]
		else:
			final_boxes, final_scores, final_cls_inds = np.empty((0, 4)), np.empty((0,)), np.empty((0,))
		return final_boxes, final_scores, final_cls_inds