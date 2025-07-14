import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load model YOLOv8 nano
model = YOLO("yolov8n.pt")  # bisa diganti dengan yolov8s.pt, yolov8m.pt, dst

# Fungsi update frame video & deteksi objek
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Jalankan deteksi
    results = model(frame)[0]

    jumlah_objek = 0

    # Gambar kotak dan label pada objek yang terdeteksi
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        jumlah_objek += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tambahkan jumlah objek ke frame
    cv2.putText(frame, f"Jumlah Objek Terdeteksi: {jumlah_objek}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Konversi ke format RGB untuk Tkinter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Panggil fungsi ini lagi setiap 10ms
    video_label.after(10, update_frame)

# ===== GUI Setup =====
window = tk.Tk()
window.title("Deteksi Semua Objek - YOLOv8 + Tkinter")
video_label = tk.Label(window)
video_label.pack()

# ===== Kamera Setup =====
cap = cv2.VideoCapture(0)  # Ubah ke 1 jika webcam eksternal

if not cap.isOpened():
    print("Kamera tidak bisa diakses.")
    exit()

# ===== Mulai Deteksi =====
update_frame()
window.mainloop()

# ===== Bersihkan =====
cap.release()
cv2.destroyAllWindows()
