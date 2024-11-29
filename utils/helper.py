import numpy as np
from PIL import Image
from ultralytics import YOLO


def get_boxes(model, path):
    results = model.predict(path, conf=0.5, iou=0.7, agnostic_nms=True, verbose=False)
    return results[0].boxes


def compute_iou(xyxy1, xyxy2):
    x1, y1 = max(xyxy1[0], xyxy2[0]), max(xyxy1[1], xyxy2[1])
    x2, y2 = min(xyxy1[2], xyxy2[2]), min(xyxy1[3], xyxy2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    area_box2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0


def combine_and_filter_boxes(boxes1, boxes2, threshold_iou=0.8):
    combined_results = {}

    for boxes in [boxes1, boxes2]:
        for i in range(len(boxes.xyxy)):
            key = tuple(boxes.xyxy[i].tolist())
            if key not in combined_results or boxes.conf[i] > combined_results[key]['conf']:
                combined_results[key] = {
                    'xyxy': boxes.xyxy[i],
                    'conf': boxes.conf[i],
                    'cls': boxes.cls[i]
                }

    filtered_results = []
    for res in combined_results.values():
        keep = True
        for fr in filtered_results:
            if compute_iou(res['xyxy'].numpy(), fr['xyxy'].numpy()) > threshold_iou:
                if res['conf'] < fr['conf']:
                    keep = False
                    break
                else:
                    filtered_results.remove(fr)
                    break
        if keep:
            filtered_results.append(res)

    return filtered_results


def crop_to_highest_container(frame, detections, is_boxes=True):
    if is_boxes:
        boxes = detections[0].boxes
        y_values = boxes.xyxy[:, 1]
    else:
        if detections.size > 0:
            y_values = detections[:, 1]
        else:
            return frame


    if len(y_values) < 1:
        return frame

    min_index = np.argmin(y_values)

    if is_boxes:
        closest_container = (boxes.xyxy[min_index]).numpy()
    else:
        closest_container = detections[min_index]

    x1, y1, x2, y2 = closest_container
    container_height = (y2 - y1)
    crop_y = y1 - (container_height // 10)

    if crop_y > 0:
        height, width, channels = frame.shape
        cropped_image = frame[int(crop_y):height, 0:width]
        return cropped_image
    else:
        return frame


def crop_to_diagonal_between_containers(frame, detections, is_boxes=True):
    if is_boxes:
        boxes = detections[0].boxes
        y_values = boxes.xyxy[:, 1]
    else:
        if detections.size > 0:
            y_values = detections[:, 1]
        else:
            return frame

    if len(y_values) < 1:
        return frame

    highest_index = np.argmax(y_values)
    lowest_index = np.argmin(y_values)

    if is_boxes:
        highest_container = (boxes.xyxy[highest_index]).numpy()
        lowest_container = (boxes.xyxy[lowest_index]).numpy()
    else:
        highest_container = detections[highest_index]
        lowest_container = detections[lowest_index]

    x1_high, y1_high, x2_high, y2_high = highest_container
    x1_low, y1_low, x2_low, y2_low = lowest_container

    height, width, _ = frame.shape

    if len(y_values) == 1:
        center_y = (y1_high + y2_high) / 2
        for y in range(height):
            for x in range(width):
                line_y = center_y
                if y < line_y:
                    frame[y, x] = (0, 0, 0)
    else:

        center_high = ((x1_high + x2_high) / 2, (y1_high + y2_high) / 2)
        center_low = ((x1_low + x2_low) / 2, (y1_low + y2_low) / 2)
        x1, y1 = center_high
        x2, y2 = center_low
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        c = y1 - m * x1

        for y in range(height):
            for x in range(width):

                if m != float('inf'):
                    line_y = m * x + c
                    if y < line_y and y < y1 or y < y2:
                        frame[y, x] = (0, 0, 0)

    return frame


def mask_rectangles(image, rectangles):
    result_image = np.zeros_like(image)

    height, width = image.shape[:2]

    for rect in rectangles:
        x1, y1, x2, y2 = map(int, rect)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        result_image[y1:y2 - int(0.8 * (y2 - y1)), x1:x2] = image[y1:y2 - int(0.8 * (y2 - y1)), x1:x2]

        area_above_y1 = max(0, y1 - (y2 - y1) // 2)
        area_above_y2 = y1

        result_image[area_above_y1:area_above_y2, x1:x2] = image[area_above_y1:area_above_y2, x1:x2]

    return result_image


def small_trash_detect(image, model, detections, is_boxes=True):
    if is_boxes:
        count_of_containers = len(detections[0].boxes.data)
    else:
        count_of_containers = detections.size
    results = model(image, conf=0.5, agnostic_nms=True, verbose=False, max_det=50)
    image = Image.fromarray(results[0].plot())
    image.show()
    num_detections = len(results[0].boxes.data)
    trash_level = num_detections / count_of_containers if count_of_containers > 0 else 0
    if trash_level < 0.1:
        return 0
    elif 0.1 <= trash_level < 0.5:
        return 1
    elif 0.5 <= trash_level < 1:
        return 2
    elif 1 <= trash_level < 1.5:
        return 3
    elif 1.5 <= trash_level < 2:
        return 4
    elif 2 <= trash_level < 2.5:
        return 5
    elif 2.5 <= trash_level < 3:
        return 6
    elif 3 <= trash_level < 4:
        return 7
    elif 4 <= trash_level < 5:
        return 8
    elif trash_level >= 5:
        return 9
    elif trash_level > 10:
        return 10


def large_trash_detect(image, model, detections, is_boxes=True):
    if is_boxes:
        count_of_containers = len(detections[0].boxes.data)
    else:
        count_of_containers = detections.size
    results = model(image, conf=0.5, agnostic_nms=True, verbose=False, max_det=50)
    num_detections = len(results[0].boxes.data)
    trash_level = num_detections / count_of_containers if count_of_containers > 0 else 0
    if trash_level < 0.1:
        return 0
    elif 0.1 <= trash_level < 0.5:
        return 1
    elif 0.5 <= trash_level < 1:
        return 2
    elif 1 <= trash_level < 1.5:
        return 3
    elif 1.5 <= trash_level < 2:
        return 4
    elif 2 <= trash_level < 2.5:
        return 5
    elif 2.5 <= trash_level < 3:
        return 6
    elif 3 <= trash_level < 4:
        return 7
    elif 4 <= trash_level < 5:
        return 8
    elif trash_level >= 5:
        return 9
    elif trash_level > 10:
        return 10


def crop_only_containers(frame, detections, is_boxes=True):
    if is_boxes:
        boxes = detections[0].boxes
        if len(boxes) < 1:
            return frame
        else:
            containers = np.array(boxes.xyxy)

    else:
        if detections.size > 0:
            containers = np.array(detections)
        else:
            return frame
    result_image = mask_rectangles(frame, containers)
    return result_image


def get_info(detections, is_boxes=True):
    if is_boxes:
        count_of_closed = (detections[0].boxes.cls.eq(0)).sum().item()
        count_of_empty = (detections[1].boxes.cls.eq(1)).sum().item()
        count_of_full = (detections[2].boxes.cls.eq(2)).sum().item()
        return count_of_closed, count_of_empty, count_of_full
    else:
        cls = np.array([res['cls'].numpy() for res in detections])
        count_of_closed = (cls == 0).sum().item()
        count_of_empty = (cls == 1).sum().item()
        count_of_full = (cls == 2).sum().item()

        return count_of_closed, count_of_empty, count_of_full


def confirmation_of_the_results(frame, results, model1, model2, is_boxes=True):
    if is_boxes:
        boxes = results[0].boxes
        if len(boxes) < 1:
            return None
        else:
            cls = np.array(boxes.cls)
            conf = np.array(boxes.conf)
            xyxy = np.array(boxes.xyxy)

    else:
        if results.size > 0:
            cls = np.array([res['cls'].numpy() for res in results])
            conf = np.array([res['conf'].numpy() for res in results])
            xyxy = np.array([res['xyxy'].numpy() for res in results])
        else:
            return None
    results_of_detection1 = model1(conf=0.85, agnostic_nms=True, verbose=False, max_det=50)
    results_of_detection2 = model2(conf=0.85, agnostic_nms=True, verbose=False, max_det=50)
    # Эту функцию надо дописать


def combine_results(model1, model2, path):
    boxes1 = get_boxes(model1, path)
    boxes2 = get_boxes(model2, path)
    return combine_and_filter_boxes(boxes1, boxes2)


def load_model(path):
    return YOLO(path)
