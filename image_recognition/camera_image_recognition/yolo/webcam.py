import os
import queue
import sys
import threading

import cv2
import numpy as np
import pygame  # gui w opencv nie pokazywało obrazu
from ultralytics.models.yolo.model import YOLO

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

CLASSES: list[str] = []
COLORS = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]  # RGB
CONF_THRESH = 0.2
IMG_SIZE = 256
INFER_EVERY = 3
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "best.pt"

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)


def inference_worker():
    global CLASSES
    m = YOLO(MODEL_PATH)
    CLASSES = [m.names[i] for i in range(len(m.names))]

    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    m(dummy, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
    print(f"Model załadowany! Klasy: {CLASSES}")

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        results = m(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

        boxes  = results.boxes.xyxy.cpu().tolist()
        labels = results.boxes.cls.cpu().tolist()
        scores = results.boxes.conf.cpu().tolist()

        detections = sorted(zip(boxes, labels, scores), key=lambda x: x[2], reverse=True)

        try:
            result_queue.put_nowait(detections)
        except queue.Full:
            pass


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nie można otworzyć kamery")
    return cap


def init_display(width: int, height: int) -> tuple:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("YOLOv26n (q = wyjście)")
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()
    return screen, font, clock


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            raise SystemExit


def maybe_enqueue_frame(frame, frame_idx: int):
    if frame_idx % INFER_EVERY == 0:
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass


def poll_detections(last_detections: list, model_ready: bool) -> tuple[list, bool]:
    try:
        last_detections = result_queue.get_nowait()
        model_ready = True
    except queue.Empty:
        pass
    return last_detections, model_ready


def draw_frame(screen, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))


def draw_loading(screen, font):
    txt = font.render("Ladowanie modelu...", True, (0, 200, 255))
    screen.blit(txt, (20, 20))


def draw_detections(screen, font, detections: list):
    visible = 0
    for box, label, score in detections:
        cls_idx = int(label)
        if cls_idx >= len(CLASSES):
            continue
        visible += 1
        x1, y1, x2, y2 = (int(v) for v in box)
        color = COLORS[cls_idx % len(COLORS)]
        pygame.draw.rect(screen, color, (x1, y1, x2 - x1, y2 - y1), 2)
        label_txt = font.render(f"{CLASSES[cls_idx]} {score:.2f}", True, color)
        screen.blit(label_txt, (x1, max(y1 - 20, 0)))

    det_txt = font.render(f"Detekcje: {visible}", True, (255, 255, 255))
    screen.blit(det_txt, (20, 20))


def main():
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    cap = open_camera()
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    screen, font, clock = init_display(cam_w, cam_h)

    last_detections: list = []
    model_ready = False
    frame_idx = 0

    try:
        while True:
            handle_events()

            ret, frame = cap.read()
            if not ret:
                continue

            maybe_enqueue_frame(frame, frame_idx)
            last_detections, model_ready = poll_detections(last_detections, model_ready)

            draw_frame(screen, frame)

            if not model_ready:
                draw_loading(screen, font)
            else:
                draw_detections(screen, font, last_detections)

            pygame.display.flip()
            clock.tick(30)
            frame_idx += 1

    except SystemExit:
        pass
    finally:
        frame_queue.put(None)
        cap.release()
        pygame.quit()


if __name__ == "__main__":
    main()
