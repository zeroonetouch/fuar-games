import cv2
import numpy as np
from ultralytics import YOLO
import random
import time


class LogoInteractionGame:
    def __init__(self, model_path='yolov8m-pose.pt', logo_path='zot_logo.jpeg', overlay_path='./pose-game/pose-game.png', width=3840, height=2160, game_duration=30, collision_threshold=100, logo_size_ratio=0.2):
        # YOLO v8-pose modelini yükle
        self.model = YOLO(model_path)

        # Sol ve sağ el bilekleri için renkler
        self.left_wrist_color = (216, 166, 1)   # Yeşil
        self.right_wrist_color = (144, 51, 98)  # Kırmızı

        # Logo dosyasını yükle ve yuvarlak hale getir
        self.logo = self.circular_crop(cv2.imread(logo_path, cv2.IMREAD_UNCHANGED))
        self.logo = self.resize_logo(self.logo, logo_size_ratio)  # Logoyu küçült

        # Pose-game overlay görselini yükle
        self.overlay_image = cv2.imread(overlay_path)

        # Ekran boyutları
        self.height = height
        self.width = width
        self.logo_radius = self.logo.shape[0] // 2
        self.collision_threshold = collision_threshold

        # Logo için rastgele başlangıç pozisyonu
        self.x_logo = random.randint(self.logo_radius, self.width - self.logo_radius)
        self.y_logo = random.randint(self.logo_radius, self.height - self.logo_radius)

        # Oyun durumu
        self.score = 0
        self.game_started = False
        self.start_time = time.time()
        self.instruction_duration = 5  # Bilgilendirme süresi (saniye)
        self.game_duration = game_duration  # Oyun süresi (saniye)
        self.circle_radius = 100  # Daire çapı
        self.opacity = 0.3

        # Webcam'i başlat
        self.cap = cv2.VideoCapture(1)

    def circular_crop(self, image):
        size = min(image.shape[0], image.shape[1])
        x_center = image.shape[1] // 2
        y_center = image.shape[0] // 2
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2, (255, 255, 255), -1)
        circular_image = np.zeros((size, size, 4), np.uint8)
        circular_image[:, :, :3] = image[y_center-size//2:y_center+size//2, x_center-size//2:x_center+size//2, :3]
        circular_image[:, :, 3] = mask
        return circular_image

    def resize_logo(self, logo, scale_factor):
        new_size = int(logo.shape[1] * scale_factor)
        return cv2.resize(logo, (new_size, new_size), interpolation=cv2.INTER_AREA)

    def display_logo(self, frame):
        overlay = frame.copy()

        # Logoyu ekrana yerleştirirken alpha blending yap
        for i in range(self.logo.shape[0]):
            for j in range(self.logo.shape[1]):
                if self.logo[i, j, 3] != 0:  # Alpha kanalı
                    # Mevcut çerçevedeki piksel
                    background_pixel = frame[self.y_logo - self.logo_radius + i, self.x_logo - self.logo_radius + j]
                    # Logodaki piksel
                    logo_pixel = self.logo[i, j, :3]

                    # Opaklığı logo_opacity ile ayarla
                    alpha = 0.8 * (self.logo[i, j, 3] / 255.0)  # Alfa kanalını kullanarak opaklık hesapla
                    blended_pixel = (1 - alpha) * background_pixel + alpha * logo_pixel

                    # Sonucu çerçeveye yerleştir
                    frame[self.y_logo - self.logo_radius + i, self.x_logo - self.logo_radius + j] = blended_pixel

    def check_collision(self, x_wrist, y_wrist):
        # Çarpışma toleransını ekleyelim
        return (x_wrist - self.x_logo)**2 + (y_wrist - self.y_logo)**2 < (self.logo_radius + self.collision_threshold)**2

    def display_text(self, frame, text, position, scale=2.0, color=(255, 255, 255), thickness=3):
        # Metin arkasına şeffaf dikdörtgen
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, scale, thickness)
        x, y = position
        background_start = (x, y - text_height - 10)
        background_end = (x + text_width + 10, y + 10)
        overlay = frame.copy()
        cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Metni ekle
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_COMPLEX, scale, color, thickness)

    def run_game(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Çerçeveyi boyutlandır
            frame = cv2.resize(frame, (self.width, self.height))
            overlay = frame.copy()
            elapsed_time = time.time() - self.start_time

            if not self.game_started:
                countdown_text = f"Basliyor: {int(self.instruction_duration - elapsed_time)} saniye"
                self.display_text(frame, countdown_text, (50, self.height // 2 + 20))

                if elapsed_time > self.instruction_duration:
                    self.game_started = True
                    self.start_time = time.time()  # Oyunun başlangıç zamanını yeniden başlat

            elif elapsed_time <= self.game_duration:
                remaining_time = self.game_duration - elapsed_time

                # YOLO modelini kullanarak tahmin yap
                results = self.model(frame, verbose=False)

                # El bileklerini tespit et
                for result in results:
                    keypoints = result.keypoints.xy
                    if keypoints is not None and keypoints.shape[1] > 0:
                        # Sol el bileği (9 numaralı keypoint)
                        left_wrist = keypoints[0][9]
                        x_left, y_left = int(left_wrist[0]), int(left_wrist[1])
                        cv2.circle(frame, (x_left, y_left), 5, self.left_wrist_color, -1)

                        # Daireyi çiz (sol bilek)
                        cv2.circle(overlay, (x_left, y_left), self.circle_radius, self.left_wrist_color, -1)

                        # Sağ el bileği (10 numaralı keypoint)
                        right_wrist = keypoints[0][10]
                        x_right, y_right = int(right_wrist[0]), int(right_wrist[1])
                        cv2.circle(frame, (x_right, y_right), 5, self.right_wrist_color, -1)

                        # Daireyi çiz (sağ bilek)
                        cv2.circle(overlay, (x_right, y_right), self.circle_radius, self.right_wrist_color, -1)

                        # Çarpışma kontrolü
                        if self.check_collision(x_left, y_left) or self.check_collision(x_right, y_right):
                            self.score += 1
                            # Yeni rastgele konum
                            self.x_logo = random.randint(self.logo_radius, self.width - self.logo_radius)
                            self.y_logo = random.randint(self.logo_radius, self.height - self.logo_radius)

                # Logoyu çerçeveye yerleştir
                self.display_logo(frame)

                # Skoru ve geri sayımı ekranda göster
                score_text = f"Skor: {self.score}"
                self.display_text(frame, score_text, (10, 80), scale=2.0)

                timer_text = f"Kalan sure: {int(remaining_time)} saniye"
                self.display_text(frame, timer_text, (10, 160), scale=2.0)

            else:
                # Oyun sona erdi, final skoru göster
                game_over_text = f"Oyun Bitti! Final Skor: {self.score}"
                self.display_text(frame, game_over_text, (50, 300))

            # Bileklerin etrafına çizilen daireyi opaklıkla birleştir
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)

            # Sağ tarafa 'pose-game.png' görselini ekle
            frame_with_overlay = cv2.hconcat([frame, self.overlay_image])

            # Sonuçları göster
            cv2.imshow("ZOT Logo Interaction", frame_with_overlay)

            # 'q' tuşuna basarak çıkış yap
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Oyun başlatma
game = LogoInteractionGame()
game.run_game()
