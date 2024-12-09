import json
import time
import cv2
from multiprocessing import Process
import numpy as np
import psycopg2
from ultralytics import YOLO
from server import serve
import os
import utils.helper as helper

def get_connection():
    dbname = os.getenv("DBNAME")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return psycopg2.connect(
        database=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
def capture_stream(video_path: str, model1_for_containers: YOLO, model2_for_containers: YOLO, model_for_small_trash: YOLO, model_for_large_trash: YOLO, time_interval: int, region_of_detection: str):

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError("Не удалось открыть видеопоток: {}".format(video_path))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  detection_coordinates = helper.str_to_coordinates(region_of_detection, h, w)

  conn = get_connection()
  cursor = conn.cursor()
  last_processed_time = time.time()
  print("Началось чтение видеопотока")

  while True:
      ret, frame = cap.read()
      if not ret:
          break
      current_time = time.time()

      if int(current_time) - int(last_processed_time) >= int(time_interval):
          last_processed_time = current_time
          frame = helper.select_area_for_detection(frame, detection_coordinates)
          filtered_results = helper.combine_results(model1_for_containers, model2_for_containers, frame)
          detections = np.array([res['xyxy'] for res in filtered_results])
          image = helper.crop_to_diagonal_between_containers(frame, detections, is_boxes=False)
          small_trash_level = helper.small_trash_detect(image, model_for_small_trash, detections, is_boxes=False)
          large_trash_level = helper.large_trash_detect(image, model_for_large_trash, detections, is_boxes=False)
          count_of_closed, count_of_empty, count_of_full = helper.get_info(filtered_results, is_boxes=False)
          cursor.execute(
              '''INSERT INTO results (full_cans, empty_cans, closed_cans, small_garbage_level, large_garbage_level, detection_time) VALUES (%s, %s, %s, %s, %s, %s)''',
              (count_of_full, count_of_empty, count_of_closed, small_trash_level, large_trash_level, current_time)
          )
          conn.commit()
      else:
          continue

  cap.release()
  cursor.close()
  conn.close()
  print("Видеопоток закрыт")






if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    connected = False
    attempts = 0

    while not connected:
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS results (
                            full_cans INT,
                            empty_cans INT,
                            closed_cans INT,
                            small_garbage_level INT,
                            large_garbage_level INT,
                            detection_time INT
                        );
                    ''')

                    conn.commit()
                    print("Подключение к TimescaleDB прошло успешно!")
                    connected = True

        except Exception as e:
            attempts += 1
            print(f"Попытка {attempts}: Подключение к TimescaleDB не удалось. Ошибка: {e}")
            time.sleep(10)




    video_path = os.getenv("VIDEO_PATH")

    model1_for_containers = YOLO(config['models']['model1_for_containers'])
    model2_for_containers = YOLO(config['models']['model2_for_containers'])
    model_for_small_trash = YOLO(config['models']['model_for_small_trash'])
    model_for_large_trash = YOLO(config['models']['model_for_large_trash'])
    time_interval = os.getenv("CHECK_TIME_INTERVAL")
    region_of_detection = os.getenv("REGION_OF_DETECTION")

    capture_stream_process = Process(target=capture_stream, args=(video_path, model1_for_containers, model2_for_containers, model_for_small_trash, model_for_large_trash, time_interval, region_of_detection, ))
    capture_stream_process.daemon = False
    capture_stream_process.start()
    server_process = Process(target=serve, args=())
    server_process.daemon = False
    server_process.start()

    while True:
        try:
            if not capture_stream_process.is_alive():
                capture_stream_process = Process(target=capture_stream, args=(video_path, model1_for_containers, model2_for_containers, model_for_small_trash, model_for_large_trash, time_interval, region_of_detection, ))
                capture_stream_process.daemon = False
                capture_stream_process.start()
            if not server_process.is_alive():
                server_process = Process(target=serve, args=())
                server_process.daemon = False
                server_process.start()
        except KeyboardInterrupt:
            print("Программа была прервана пользователем")
            exit()


