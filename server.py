import os
import grpc
from concurrent import futures
import time
import psycopg2
import garbage_and_dumpsters_data_pb2_grpc
import garbage_and_dumpsters_data_pb2


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

class DumpstersAndGarbageService(garbage_and_dumpsters_data_pb2_grpc.DumpstersAndGarbageServiceServicer):
    def GetDumpstersAndGarbage(self, request, context):
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MIN(detection_time) AS min_time FROM results")
                min_time = cursor.fetchone()[0]

                time = request.time

                if min_time is None:
                    return garbage_and_dumpsters_data_pb2.DumpstersAndGarbageData(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        small_garbage_level=0,
                        large_garbage_level=0,
                        time=0
                    )

                if time < min_time:
                    return garbage_and_dumpsters_data_pb2.DumpstersAndGarbageData(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        small_garbage_level=0,
                        large_garbage_level=0,
                        time=0
                    )

                query = """
                        SELECT
                            full_cans AS count_of_full,
                            empty_cans AS count_of_empty,
                            closed_cans AS count_of_closed,
                            small_garbage_level AS small_garbage_level,
                            large_garbage_level AS large_garbage_level,
                            detection_time AS time
                        FROM results 
                        WHERE detection_time <= %s
                        ORDER BY detection_time DESC
                        LIMIT 1
                    """
                cursor.execute(query, (time,))
                result = cursor.fetchone()

                if result is None:
                    return garbage_and_dumpsters_data_pb2.DumpstersAndGarbageData(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        small_garbage_level=0,
                        large_garbage_level=0,
                        time=0
                    )
                else:
                    return garbage_and_dumpsters_data_pb2.DumpstersAndGarbageData(
                        count_of_full=result[0],
                        count_of_empty=result[1],
                        count_of_closed=result[2],
                        small_garbage_level=result[3],
                        large_garbage_level=result[4],
                        time=result[5]
                    )
        finally:
            conn.close()


class DumpstersService(garbage_and_dumpsters_data_pb2_grpc.DumpstersServiceServicer):
    def GetDumpstersCount(self, request, context):
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MIN(detection_time) AS min_time FROM results")
                min_time = cursor.fetchone()[0]

                time = request.time

                if min_time is None:
                    return garbage_and_dumpsters_data_pb2.DumpstersCount(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        time=0
                    )

                if time < min_time:
                    return garbage_and_dumpsters_data_pb2.DumpstersCount(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        time=0
                    )

                query = """
                                SELECT
                                    full_cans AS count_of_full,
                                    empty_cans AS count_of_empty,
                                    closed_cans AS count_of_closed,
                                    detection_time AS time
                                FROM results 
                                WHERE detection_time <= %s
                                ORDER BY detection_time DESC
                                LIMIT 1
                            """
                cursor.execute(query, (time,))
                result = cursor.fetchone()

                if result is None:
                    return garbage_and_dumpsters_data_pb2.DumpstersCount(
                        count_of_full=0,
                        count_of_empty=0,
                        count_of_closed=0,
                        time=0
                    )
                else:
                    return garbage_and_dumpsters_data_pb2.DumpstersCount(
                        count_of_full=result[0],
                        count_of_empty=result[1],
                        count_of_closed=result[2],
                        time=result[3]
                    )
        finally:
            conn.close()


class GarbageService(garbage_and_dumpsters_data_pb2_grpc.GarbageServiceServicer):
    def GetGarbageLevel(self, request, context):
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MIN(detection_time) AS min_time FROM results")
                min_time = cursor.fetchone()[0]

                time = request.time

                if min_time is None:
                    return garbage_and_dumpsters_data_pb2.GarbageLevel(
                        small_garbage_level=0,
                        large_garbage_level=0
                    )

                if time < min_time:
                    return garbage_and_dumpsters_data_pb2.GarbageLevel(
                        small_garbage_level=0,
                        large_garbage_level=0
                    )

                query = """
                                SELECT
                                    small_garbage_level AS small_garbage_level,
                                    large_garbage_level AS large_garbage_level,
                                    detection_time AS time
                                FROM results 
                                WHERE detection_time <= %s
                                ORDER BY detection_time DESC
                                LIMIT 1
                            """
                cursor.execute(query, (time,))
                result = cursor.fetchone()

                if result is None:
                    return garbage_and_dumpsters_data_pb2.GarbageLevel(
                        small_garbage_level=0,
                        large_garbage_level=0
                    )
                else:
                    return garbage_and_dumpsters_data_pb2.GarbageLevel(
                        small_garbage_level=result[0],
                        large_garbage_level=result[1]
                    )
        finally:
            conn.close()

def serve():
    ip_address = os.getenv("GRPC_HOST")
    port = os.getenv("GRPC_PORT")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    garbage_and_dumpsters_data_pb2_grpc.add_DumpstersAndGarbageServiceServicer_to_server(DumpstersAndGarbageService(), server)
    garbage_and_dumpsters_data_pb2_grpc.add_DumpstersServiceServicer_to_server(DumpstersService(), server)
    garbage_and_dumpsters_data_pb2_grpc.add_GarbageServiceServicer_to_server(GarbageService(), server)
    server.add_insecure_port(f'{ip_address}:{port}')
    server.start()
    print(f"Сервер запущен на {ip_address}:{port}")
    try:
        while True:
            time.sleep(20)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
