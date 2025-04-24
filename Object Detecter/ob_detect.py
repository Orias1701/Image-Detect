import cv2
import numpy as np

# Đường dẫn đến file mô hình và cấu hình
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = "MobileNetSSD_deploy.prototxt"
LABELS_PATH = "labels.txt"

# Danh sách nhãn của các đối tượng
CLASSES = []
with open(LABELS_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Tải mô hình MobileNet SSD
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Giảm độ phân giải để tăng tốc
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuẩn bị frame cho mô hình
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Đưa blob vào mô hình
    net.setInput(blob)
    detections = net.forward()

    # Xử lý kết quả nhận diện
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Ngưỡng tin cậy
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Vẽ hộp và nhãn
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Object Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()