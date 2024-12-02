import json
import time
import cv2
from multiprocessing import Process
import numpy as np
from ultralytics import YOLO
from server import serve
import os
from clickhouse_driver import Client
import utils.helper as helper


def get_client():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return Client(host=host, port=port, user='default', password='', database='default')

def capture_stream(video_path: str, model1_for_containers: YOLO, model2_for_containers: YOLO, model_for_small_trash: YOLO, model_for_large_trash: YOLO, time_interval: int):

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError("Не удалось открыть видеофайл: {}".format(video_path))

  client = get_client()
  last_processed_time = time.time()
  print("Началось чтение видеопотока")

  while True:
      ret, frame = cap.read()
      if not ret:
          break
      current_time = time.time()

      if current_time - last_processed_time >= time_interval:
          last_processed_time = current_time
          filtered_results = helper.combine_results(model1_for_containers, model2_for_containers, frame)
          detections = np.array([res['xyxy'] for res in filtered_results])
          image = helper.crop_to_diagonal_between_containers(frame, detections, is_boxes=False)
          small_trash_level = helper.small_trash_detect(image, model_for_small_trash, detections, is_boxes=False)
          large_trash_level = helper.large_trash_detect(image, model_for_large_trash, detections, is_boxes=False)
          count_of_closed, count_of_empty, count_of_full = helper.get_info(filtered_results, is_boxes=False)
          client.execute(
              ''' INSERT INTO results (full_cans, empty_cans, closed_cans, small_garbage_level, large_garbage_level, detection_time) VALUES ''',
              count_of_full, count_of_empty, count_of_closed, small_trash_level, large_trash_level, current_time)
      else:
          continue

  cap.release()
  client.disconnect()
  print("Видеопоток закрыт")






if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    connected = False
    attempts = 0
    while not connected:
        try:
            client = get_client()
            client.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    full_cans Int32,
                    empty_cans Int32,
                    closed_cans Int32,
                    small_garbage_level Int32,
                    large_garbage_level Int32,
                    detection_time Int32
                ) ENGINE = MergeTree()
                ORDER BY detection_time
                ''')
            print("Подключение к ClickHouse прошло успешно!")
            connected = True
            client.disconnect()
        except Exception as e:
            attempts += 1
            print(f"Попытка {attempts}: Подключение к ClickHouse не удалось. Ошибка: {e}")
            time.sleep(10)


    video_path = os.getenv("VIDEO_PATH")

    model1_for_containers = YOLO(config['models']['model1_for_containers'])
    model2_for_containers = YOLO(config['models']['model2_for_containers'])
    model_for_small_trash = YOLO(config['models']['model_for_small_trash'])
    model_for_large_trash = YOLO(config['models']['model_for_large_trash'])
    time_interval = os.getenv("CHECK_TIME_INTERVAL")

    capture_stream_process = Process(target=capture_stream, args=(video_path, model1_for_containers, model2_for_containers, model_for_small_trash, model_for_large_trash, time_interval,))
    capture_stream_process.daemon = False
    capture_stream_process.start()
    server_process = Process(target=serve, args=())
    server_process.daemon = False
    server_process.start()

    while True:
        try:
            if not capture_stream_process.is_alive():
                capture_stream_process = Process(target=capture_stream, args=(video_path, model1_for_containers, model2_for_containers, model_for_small_trash, model_for_large_trash, time_interval,))
                capture_stream_process.daemon = False
                capture_stream_process.start()
            if not server_process.is_alive():
                server_process = Process(target=serve, args=())
                server_process.daemon = False
                server_process.start()
        except KeyboardInterrupt:
            print("Программа была прервана пользователем")
            exit()


