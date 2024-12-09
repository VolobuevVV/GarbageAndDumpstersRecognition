import math
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO

def select_area_for_detection(frame, coordinates):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(frame_rgb)
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)

    draw.polygon(coordinates, fill=255)

    result_image = Image.new('RGB', image.size)
    result_image.paste(image, mask=mask)

    draw_result = ImageDraw.Draw(result_image)
    draw_result.rectangle([0, 0, image.width, image.height], fill='black')
    result_image.paste(image, mask=mask)

    return np.array(result_image)

def str_to_coordinates(string, h, w):
    result = [(0, 0), (w, 0), (w, h), (0, h)]
    coordinates_str = string.replace('h', str(h))
    coordinates_str = coordinates_str.replace('w', str(w))

    coordinates_str = coordinates_str.strip("[]").replace(" ", "")

    if not coordinates_str:
        print("Ошибка: область детекции пустая / задана некорректно!")
        return result

    coordinate_pairs = coordinates_str.split("),(")

    coordinates = []

    for pair in coordinate_pairs:
        pair = pair.strip("()")

        if not pair:
            print("Ошибка: область детекции пустая!")
            return result

        parts = pair.split(",")

        if len(parts) != 2:
            print("Ошибка: область детекции задана некорректно!")
            return result

        try:
            x = int(eval(parts[0]))
            y = int(eval(parts[1]))

            if not (0 <= x <= w) or not (0 <= y <= h):
                print(f"Ошибка: координаты ({x}, {y}) области детекции выходят за пределы допустимых значений.")
                return result
            coordinates.append((x, y))
        except ZeroDivisionError:
            print("Ошибка: деление на ноль при задании области детекции!")
            return result

        except (ValueError, SyntaxError) as e:
            print("Ошибка: область детекции задана некорректно!")
            return result

    if len(coordinates) < 3:
        print("Ошибка: область детекции не может быть точкой или линией!")
        return result
    return coordinates

def get_boxes(model, path):
    results = model.predict(path, conf=0.7, agnostic_nms=True, verbose=False)
    boxes = results[0].boxes

    xyxy_array = boxes.xyxy.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()
    cls_array = boxes.cls.cpu().numpy()

    return xyxy_array, conf_array, cls_array


def compute_iou(xyxy1, xyxy2):
    x1, y1 = max(xyxy1[0], xyxy2[0]), max(xyxy1[1], xyxy2[1])
    x2, y2 = min(xyxy1[2], xyxy2[2]), min(xyxy1[3], xyxy2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    area_box2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0


def combine_and_filter_boxes(xyxy1, conf1, cls1, xyxy2, conf2, cls2, threshold_iou=0.7):
    combined_results = {}

    for i in range(len(xyxy1)):
        key = tuple(xyxy1[i].tolist())
        if key not in combined_results or conf1[i] > combined_results[key]['conf']:
            combined_results[key] = {
                'xyxy': xyxy1[i],
                'conf': conf1[i],
                'cls': cls1[i]
            }

    for i in range(len(xyxy2)):
        key = tuple(xyxy2[i].tolist())
        if key not in combined_results or conf2[i] > combined_results[key]['conf']:
            combined_results[key] = {
                'xyxy': xyxy2[i],
                'conf': conf2[i],
                'cls': cls2[i]
            }

    filtered_results = []
    for res in combined_results.values():
        keep = True
        for fr in filtered_results:
            if compute_iou(res['xyxy'], fr['xyxy']) > threshold_iou:
                if res['conf'] < fr['conf']:
                    keep = False
                    break
                else:
                    filtered_results = [f for f in filtered_results if not np.array_equal(f['xyxy'], fr['xyxy'])]
                    break
        if keep:
            filtered_results.append(res)


    return filtered_results



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

def calculate_average_size(detections):
    if len(detections) == 0:
        return 0
    sizes = [(det[2] - det[0]) * (det[3] - det[1]) for det in detections]
    return np.mean(sizes)


def small_trash_detect(image, model, detections, is_boxes=True):
    if is_boxes:
        count_of_containers = len(detections[0].boxes.data)
        container_sizes = [det[0] for det in detections[0].boxes.data]
    else:
        count_of_containers = detections.shape[0]
        container_sizes = detections
    if count_of_containers > 0:
        results = model(image, conf=0.5, agnostic_nms=True, verbose=False, max_det=50)
        num_detections = len(results[0].boxes.data)

        trash_detections = results[0].boxes.data[:, :4].tolist()
        average_trash_size = calculate_average_size(trash_detections)
        average_container_size = calculate_average_size(container_sizes)
        size_factor = average_trash_size / (average_container_size if average_container_size > 0 else 1)
        trash_level = (num_detections / count_of_containers) * math.log(1 + num_detections) * size_factor * 5
    else:
        height, width = image.shape[:2]
        half_area = (height // 2) * width
        black_image = np.zeros((height // 2, width, 3), dtype=np.uint8)
        combined_image = np.vstack((black_image, image[height // 2:]))
        results = model(combined_image, conf=0.5, agnostic_nms=True, verbose=False, max_det=50)
        num_detections = len(results[0].boxes.data)
        trash_detections = results[0].boxes.data[:, :4].tolist()
        total_trash_area = sum([(det[2] - det[0]) * (det[3] - det[1]) for det in trash_detections])
        trash_level = total_trash_area / half_area * math.log(1 + num_detections) * 20

    return min(10, math.ceil(trash_level))

def large_trash_detect(image, model, detections, is_boxes=True):
    if is_boxes:
        count_of_containers = len(detections[0].boxes.data)
        container_sizes = [det[0] for det in detections[0].boxes.data]
    else:
        count_of_containers = detections.shape[0]
        container_sizes = detections

    if count_of_containers > 0:
        results = model(image, conf=0.7, agnostic_nms=True, verbose=False, max_det=50)
        num_detections = len(results[0].boxes.data)

        trash_detections = results[0].boxes.data[:, :4].tolist()
        average_trash_size = calculate_average_size(trash_detections)
        average_container_size = calculate_average_size(container_sizes)
        size_factor = average_trash_size / (average_container_size if average_container_size > 0 else 1)
        trash_level = (num_detections / count_of_containers) * math.log(math.e + num_detections) * size_factor * 10
    else:
        height, width = image.shape[:2]
        half_area = (height // 2) * width
        black_image = np.zeros((height // 2, width, 3), dtype=np.uint8)
        combined_image = np.vstack((black_image, image[height // 2:]))
        results = model(combined_image, conf=0.7, agnostic_nms=True, verbose=False, max_det=50)
        num_detections = len(results[0].boxes.data)
        trash_detections = results[0].boxes.data[:, :4].tolist()
        total_trash_area = sum([(det[2] - det[0]) * (det[3] - det[1]) for det in trash_detections])
        trash_level = total_trash_area / half_area * math.log(1 + num_detections) * 20

    return min(10, math.ceil(trash_level))



def get_info(detections, is_boxes=True):
    if is_boxes:
        count_of_closed = (detections[0].boxes.cls.eq(0)).sum().item()
        count_of_empty = (detections[1].boxes.cls.eq(1)).sum().item()
        count_of_full = (detections[2].boxes.cls.eq(2)).sum().item()
        return count_of_closed, count_of_empty, count_of_full
    else:
        cls = np.array([res['cls'] for res in detections])
        count_of_closed = (cls == 0).sum().item()
        count_of_empty = (cls == 1).sum().item()
        count_of_full = (cls == 2).sum().item()

        return count_of_closed, count_of_empty, count_of_full


def combine_results(model1, model2, path):
    xyxy1, conf1, cls1 = get_boxes(model1, path)
    xyxy2, conf2, cls2 = get_boxes(model2, path)
    return combine_and_filter_boxes(xyxy1, conf1, cls1, xyxy2, conf2, cls2)


def load_model(path):
    return YOLO(path)
