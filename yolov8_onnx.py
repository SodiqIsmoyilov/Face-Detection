import onnxruntime
import numpy as np
import cv2



opt_session = onnxruntime.SessionOptions()
opt_session.enable_mem_pattern = True
opt_session.enable_cpu_mem_arena = True
opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
model_path = "models/best.onnx"
EP_list = ['CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

model_inputs = ort_session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
input_shape = model_inputs[0].shape
model_output = ort_session.get_outputs()
output_names = [model_output[i].name for i in range(len(model_output))]


def image_preprocessing(input_shape, image):
    height, width = input_shape[2:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    input_tensor = image[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor, height, width


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(2, xmax - xmin) * np.maximum(2, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

CLASSES=["Face"]
conf_thresold = 0.8

video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = video.read()

    input_tensor, h, w = image_preprocessing(input_shape, frame)
    output = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

    predictions = np.squeeze(output).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = predictions[:, :4]
    input_shape = np.array([w, h, w, h])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([width, height, width, height])
    boxes = boxes.astype(np.int32)

    indices = nms(boxes, scores, 0.3)

    image_draw = frame.copy()
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        color = (0, 255, 0)
        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)

    cv2.imshow("input", image_draw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()