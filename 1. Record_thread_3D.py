import usb.core
import usb.util
import time
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import cv2
import threading
import queue
import ffmpeg  # 설치 필요
import os

# 설정 변수
data_num = 100
data_length = 1200
fps = 30

# 실행 중인 스크립트의 경로를 기준으로 data 폴더 설정
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리
output_dir = os.path.join(script_dir, 'data', 'cms50')  # data/cms50 폴더 경로 생성

# 경로가 없으면 생성
if not os.path.exists(output_dir):
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir)  # 경로 생성
else:
    print(f"Directory exists: {output_dir}")

# 파일 경로 설정
video_d = os.path.join(output_dir, f'{data_num}_depth.mkv')
video_ir_path = os.path.join(output_dir, f'{data_num}_ir.mp4')
video_rgb_path = os.path.join(output_dir, f'{data_num}_rgb.mp4')
ppg_file_path = os.path.join(output_dir, f'{data_num}_ppg.txt')
time_file_path = os.path.join(output_dir, f'{data_num}_time.txt')

# PPG 센서 thread
def sensor_thread(dev, ep, data_queue, stop_event):
    start0 = time.time()   # 시작 시간
    data_queue.put((0.0, 0.0))  # 초기 데이터 (경과 시간 0.0으로 설정)
    while not stop_event.is_set():
        try:
            current_time = time.time()  # 현재 시간
            elapsed_time = current_time - start0  # 경과 시간 계산

            data = dev.read(ep.bEndpointAddress, ep.wMaxPacketSize)

            if data[21] != 0:  # 21번째 값이 0이 아닌 경우에만 저장
                data_queue.put((elapsed_time, data[21]))  # 경과 시간과 데이터 저장
                print(f"{elapsed_time} : {data[21]}")

        except Exception as e:
            print("[SensorThread] 에러:", e)
            break

# PPG 센서 초기화
dev = usb.core.find(idVendor=0x28E9, idProduct=0x028A)
if dev is None:
    raise ValueError('장치를 찾을 수 없습니다!')
cfg = dev.get_active_configuration()
intf = cfg[(0, 0)]
ep = usb.util.find_descriptor(
    intf,
    custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
)

# Kinect 초기화
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_2160P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=False,
    )
)
k4a.start()
k4a.whitebalance = 4500
assert k4a.whitebalance == 4500
k4a.whitebalance = 4510
assert k4a.whitebalance == 4510

# Depth 비디오 설정
input_kwargs = {
    'format': 'rawvideo',
    'pix_fmt': 'gray16le',
    's': f'{640}x{576}',
    'r': str(30)
}
process_d = (
    ffmpeg
    .input('pipe:', **input_kwargs)
    .output(video_d, vcodec='ffv1', pix_fmt='gray16le', r=30)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

# IR 및 RGB 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_ir = cv2.VideoWriter(video_ir_path, fourcc, fps, (640, 576), isColor=False)
video_rgb = cv2.VideoWriter(video_rgb_path, fourcc, fps, (3840, 2160), isColor=True)

# PPG 센서 스레드 준비
assert ep is not None
sensor_data_queue = queue.Queue()
stop_event = threading.Event()
ppg_thread = threading.Thread(target=sensor_thread, args=(dev, ep, sensor_data_queue, stop_event))
ppg_thread.daemon = True
ppg_thread.start()

# 시간 데이터
num = 0
time_data = np.zeros(data_length)

# 동기화 시작 시간
start_time = time.time()

while True:
    capture = k4a.get_capture()
    if np.any(capture.ir) and np.any(capture.color):
        elapsed_time = time.time() - start_time
        time_data[num] = elapsed_time  # 경과 시간 저장

        # 비디오 생성
        depth = capture.depth
        ir = capture.ir
        rgb = capture.color

        process_d.stdin.write(depth.tobytes())  # Depth 이미지 쓰기

        clip_value = np.percentile(ir, 99)
        ir_new = np.clip(ir, 0, 3600)
        ir_new = cv2.normalize(ir_new * 5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        video_ir.write(ir_new)  # IR 이미지 쓰기

        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)  # BGRA -> BGR 변환
            video_rgb.write(rgb)
        else:
            print("No RGB Data")
            continue

        if num == data_length - 1:
            stop_event.set()
            ppg_thread.join()
            sensor_data_list = []
            while not sensor_data_queue.empty():
                sensor_data_list.append(sensor_data_queue.get())
            print(len(sensor_data_list))
            print(sensor_data_list[0])

            ppg_data = np.array(sensor_data_list)
            np.savetxt(ppg_file_path, ppg_data, fmt='%.7f')  # PPG 데이터 저장
            np.savetxt(time_file_path, time_data, fmt='%.5f')  # 시간 데이터 저장

            video_ir.release()  # IR 비디오
            process_d.stdin.close()  # Depth 비디오
            video_rgb.release()  # RGB 비디오

            process_d.wait()
            break
        num += 1
