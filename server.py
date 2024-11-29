import os
import grpc
from concurrent import futures
import time
from clickhouse_driver import Client
import garbage_and_dumpsters_data_pb2_grpc
import garbage_and_dumpsters_data_pb2



def get_client():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return Client(host=host, port=port, user='default', password='', database='default')

class DumpstersService(garbage_and_dumpsters_data_pb2_grpc.DumpstersServiceServicer):
    def GetDumpstersCount(self, request, context):
        client = get_client()

        min_time_query = "SELECT MIN(detection_time) AS min_time FROM results"
        min_time_result = client.execute(min_time_query)

        min_time = min_time_result[0][0]
        time = request.time

        if time < min_time:
            return garbage_and_dumpsters_data_pb2.DumpstersCount(
                count_of_full=0,
                count_of_empty=0,
                count_of_closed=0
            )
        else:
            query = f"""
                       SELECT
                       count_of_full AS count_of_full,
                       count_of_empty AS count_of_empty,
                       count_of_closed AS count_of_closed
                       FROM results 
                       WHERE detection_time <= '{time}'
                       ORDER BY detection_time DESC
                       LIMIT 1 
                   """

        result = client.execute(query)[0]
        client.disconnect()

        if result is None:
            return garbage_and_dumpsters_data_pb2.DumpstersCount(
                count_of_full=0,
                count_of_empty=0,
                count_of_closed=0
            )
        else:
            return garbage_and_dumpsters_data_pb2.DumpstersCount(
                count_of_full=result[0],
                count_of_empty=result[1],
                count_of_closed=result[2]
            )


class SmallGarbageService(garbage_and_dumpsters_data_pb2_grpc.SmallGarbageServiceServicer):
    def GetSmallGarbageLevel(self, request, context):
        client = get_client()

        min_time_query = "SELECT MIN(detection_time) AS min_time FROM results"
        min_time_result = client.execute(min_time_query)

        min_time = min_time_result[0][0]
        time = request.time

        if time < min_time:
            return garbage_and_dumpsters_data_pb2.SmallGarbageLevel(
                small_garbage_level=0

            )
        else:
            query = f"""
                        SELECT
                        small_garbage_level AS small_garbage_level,
                        FROM results 
                        WHERE detection_time <= '{time}'
                        ORDER BY detection_time DESC
                        LIMIT 1 
                    """

        result = client.execute(query)[0]
        client.disconnect()

        if result is None:
            return garbage_and_dumpsters_data_pb2.SmallGarbageLevel(
                small_garbage_level=0
            )
        else:
            return garbage_and_dumpsters_data_pb2.SmallGarbageLevel(
                small_garbage_level=result[0]
            )

class LargeGarbageService(garbage_and_dumpsters_data_pb2_grpc.LargeGarbageServiceServicer):
    def GetLargeGarbageLevel(self, request, context):
        client = get_client()

        min_time_query = "SELECT MIN(detection_time) AS min_time FROM results"
        min_time_result = client.execute(min_time_query)

        min_time = min_time_result[0][0]
        time = request.time

        if time < min_time:
            return garbage_and_dumpsters_data_pb2.LargeGarbageLevel(
                large_garbage_level=0

            )
        else:
            query = f"""
                        SELECT
                        large_garbage_level AS large_garbage_level,
                        FROM results 
                        WHERE detection_time <= '{time}'
                        ORDER BY detection_time DESC
                        LIMIT 1 
                    """

        result = client.execute(query)[0]
        client.disconnect()

        if result is None:
            return garbage_and_dumpsters_data_pb2.LargeGarbageLevel(
                large_garbage_level=0
            )
        else:
            return garbage_and_dumpsters_data_pb2.LargeGarbageLevel(
                large_garbage_level=result[0]
            )

def serve():
    ip_address = os.getenv("GRPC_HOST")
    port = os.getenv("GRPC_PORT")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    garbage_and_dumpsters_data_pb2_grpc.add_DumpstersServiceServicer_to_server(DumpstersService(), server)
    garbage_and_dumpsters_data_pb2_grpc.add_SmallGarbageServiceServicer_to_server(SmallGarbageService(), server)
    garbage_and_dumpsters_data_pb2_grpc.add_LargeGarbageServiceServicer_to_server(LargeGarbageService(), server)
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
