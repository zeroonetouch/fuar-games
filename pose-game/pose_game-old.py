import cv2
import numpy as np
from ultralytics import YOLO
import random
import time

# YOLO v8-pose modelini yükle
model = YOLO('yolov8m-pose.pt')

# Sol ve sağ el bilekleri için renkler
left_wrist_color = (0, 255, 0)   # Yeşil
right_wrist_color = (0, 0, 255)  # Kırmızı

# Logo dosyasını yükle
logo = cv2.imread('zot_logo.jpeg', cv2.IMREAD_UNCHANGED)

# Logoyu yuvarlak formata dönüştürme


def circular_crop(image):
    size = min(image.shape[0], image.shape[1])
    x_center = image.shape[1] // 2
    y_center = image.shape[0] // 2
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2, (255, 255, 255), -1)
    circular_image = np.zeros((size, size, 4), np.uint8)
    circular_image[:, :, :3] = image[y_center-size//2:y_center+size//2, x_center-size//2:x_center+size//2, :3]
    circular_image[:, :, 3] = mask
    return circular_image


# Logoyu yuvarlak formata dönüştür
logo_circular = circular_crop(logo)

# Logo boyutunu küçültme (scale_factor belirleyerek)
scale_factor = 0.1  # Logonun boyutunu %50'ye küçült
new_size = int(logo_circular.shape[1] * scale_factor)
logo_circular = cv2.resize(logo_circular, (new_size, new_size), interpolation=cv2.INTER_AREA)

# Ekran boyutları
height, width = 720, 1280  # Ekran boyutu
logo_radius = logo_circular.shape[0] // 2

# Logo boyutunu kontrol et
if logo_radius >= width // 2 or logo_radius >= height // 2:
    raise ValueError("Logo boyutu ekran boyutuna göre çok büyük. Lütfen logo boyutunu küçültün.")

# Logo için başlangıç pozisyonu
x_logo = random.randint(logo_radius, width - logo_radius)
y_logo = random.randint(logo_radius, height - logo_radius)

# Skor ve oyun durumu
score = 0
game_started = False
start_time = time.time()
instruction_duration = 5  # Bilgilendirme süresi (saniye)
game_duration = 30  # Oyun süresi (saniye)
countdown_start = instruction_duration

# Kare boyutu ve opaklık
square_size = 50
opacity = 0.3

# Webcam'i aç
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Çerçeveyi boyutlandır
    frame = cv2.resize(frame, (width, height))

    overlay = frame.copy()

    elapsed_time = time.time() - start_time

    if not game_started:
        instruction_text = "Baloncuklari el bileklerinizle yakalayin!"
        cv2.putText(frame, instruction_text, (50, height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        countdown_text = f"Baslama: {int(instruction_duration - elapsed_time)} saniye"
        cv2.putText(frame, countdown_text, (50, height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if elapsed_time > instruction_duration:
            game_started = True
            start_time = time.time()  # Oyunun başlangıç zamanını yeniden başlat

    elif elapsed_time <= game_duration:
        remaining_time = game_duration - elapsed_time

        # YOLO modelini kullanarak tahmin yap
        results = model(frame)

        # El bileklerini tespit et
        for result in results:
            keypoints = result.keypoints.xy
            if keypoints is not None and keypoints.shape[1] > 0:
                # Sol el bileği (9 numaralı keypoint)
                left_wrist = keypoints[0][9]
                x_left, y_left = int(left_wrist[0]), int(left_wrist[1])
                cv2.circle(frame, (x_left, y_left), 5, left_wrist_color, -1)

                # Kareyi çiz (sol bilek)
                cv2.rectangle(overlay, (x_left - square_size // 2, y_left - square_size // 2),
                              (x_left + square_size // 2, y_left + square_size // 2),
                              left_wrist_color, -1)

                # Sağ el bileği (10 numaralı keypoint)
                right_wrist = keypoints[0][10]
                x_right, y_right = int(right_wrist[0]), int(right_wrist[1])
                cv2.circle(frame, (x_right, y_right), 5, right_wrist_color, -1)

                # Kareyi çiz (sağ bilek)
                cv2.rectangle(overlay, (x_right - square_size // 2, y_right - square_size // 2),
                              (x_right + square_size // 2, y_right + square_size // 2),
                              right_wrist_color, -1)

                # Çarpışma kontrolü
                if (x_left - x_logo)**2 + (y_left - y_logo)**2 < logo_radius**2 or \
                   (x_right - x_logo)**2 + (y_right - y_logo)**2 < logo_radius**2:
                    # Skoru artır
                    score += 1
                    # Yeni rastgele konum
                    x_logo = random.randint(logo_radius, width - logo_radius)
                    y_logo = random.randint(logo_radius, height - logo_radius)

        # Logoyu çerçeveye yerleştir
        for i in range(logo_circular.shape[0]):
            for j in range(logo_circular.shape[1]):
                if logo_circular[i, j, 3] != 0:  # Alpha kanalı
                    frame[y_logo - logo_radius + i, x_logo - logo_radius + j] = logo_circular[i, j, :3]

        # Skoru ve geri sayımı ekranda göster
        score_text = f"Skor: {score}"
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        timer_text = f"Kalan sure: {int(remaining_time)} saniye"
        cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        # Oyun sona erdi, final skoru göster
        game_over_text = f"Oyun Bitti! Final Skor: {score}"
        cv2.putText(frame, game_over_text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Bileklerin etrafına çizilen kareyi opaklıkla birleştir
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    # Sonuçları göster
    cv2.imshow("YOLO Pose Detection - Logo Interaction", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
