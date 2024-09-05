import datetime
import threading
import cv2
from face_detector import extract_face_boxes
from predict_emotion_df import detect_emotion_df
from predict_emotion_fer import detect_emotion_fer
import time
import random
from ultralytics import YOLO


class EmotionMimicGame:
    def __init__(self, emotion_model='fer', total_rounds=5, round_duration=10, countdown_duration=3, min_face_size=10, width=3840, height=2160):

        self.player_color_palette = [
            (255, 0, 0),       # Red
            (0, 255, 0),       # Green
            (0, 0, 255),       # Blue
            (255, 255, 0),     # Yellow
            (255, 165, 0),     # Orange
            (128, 0, 128),     # Purple
            (0, 255, 255),     # Cyan
            (255, 192, 203),   # Pink
            (139, 69, 19),     # Brown
            (0, 128, 0)        # Dark Green
        ]

        self.emotion_color_map = {
            'happy': (255, 255, 153),      # Daha açık sarı
            'sad': (153, 204, 255),        # Daha açık mavi
            'angry': (255, 102, 102),      # Daha açık kırmızı
            'surprise': (255, 178, 102),   # Daha açık turuncu
            'fear': (178, 102, 255),       # Daha açık mor
            'neutral': (153, 255, 255)     # Daha açık camgöbeği
        }

        self.model = YOLO('./emotion-game/yolov10b-face.pt', verbose=False)
        self.emotion_model = emotion_model
        self.total_rounds = total_rounds
        self.round_duration = round_duration
        self.countdown_duration = countdown_duration
        self.min_face_size = min_face_size
        self.width = width
        self.height = height

        self.emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral']
        self.resolution_scale = 0.5
        self.color_white = (255, 255, 255)
        self.color_yellow = (0, 255, 255)
        self.player_colors = {}
        self.overall_player_scores = {}
        self.side_image = cv2.imread('./emotion-game/emotion-mimic-side.png')
        self.original_scor_bg_image = cv2.imread('./emotion-game/score-bg.png')
        self.scor_bg_image = self.original_scor_bg_image.copy()

        self._is_disposed = False
        self.__capture = None
        self.frame = None
        self.frame_read_lock = threading.Lock()

    def assign_random_color(self):
        # Renk paletinden rastgele bir renk seç
        return random.choice(self.player_color_palette)

    def detect_faces(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)[0]
        predictions = extract_face_boxes(results, frame)
        faces, labels, ids = [], [], []

        for box in results.boxes:
            if box.id is not None:
                ids.append(int(box.id[0]))

        for row in predictions:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            width = x2 - x1
            height = y2 - y1

            if width >= self.min_face_size and height >= self.min_face_size:
                face = frame[y1:y2, x1:x2]
                faces.append(face)
                labels.append((x1, y1, x2, y2))

        return faces, labels, ids

    def detect_emotions(self, face):
        if self.emotion_model == 'fer':
            return detect_emotion_fer(face)
        else:
            return detect_emotion_df(face)

    def put_text(self, frame, text, position, color, thickness_scale=2e-3, font_scale=2.0, font=cv2.FONT_HERSHEY_COMPLEX):
        # Şık bir yazı tipi ve boyut ayarı ekleyelim
        scale = font_scale
        thickness = int(thickness_scale * min(frame.shape[1], frame.shape[0]))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        if x + text_width > frame.shape[1]:
            x = frame.shape[1] - text_width
        if y - text_height < 0:
            y = text_height + baseline
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def display_frame(self, frame, wait=True, wait_key=1):
        self.add_scores_text()
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.hconcat([frame, self.side_image])
        final_frame = cv2.hconcat([self.scor_bg_image, frame])
        cv2.imshow('Emotion Mimic Game', final_frame)
        if wait:
            cv2.waitKey(wait_key)
        else:
            cv2.waitKey(1)
        self.reset_score_bg()

    def add_scores_text(self):
        if len(self.overall_player_scores) == 0:
            return
        y_offset = 500
        for player_id, score in self.overall_player_scores.items():
            color = self.player_colors[player_id]
            self.put_text(self.scor_bg_image, f"Player {player_id} Score: {
                          score:.2f}", (50, y_offset), color, font_scale=2.0, font=cv2.FONT_HERSHEY_COMPLEX)
            y_offset += 100

    def display_target_emotion(self, frame, target_emotion):
        # Hedef duygu ismini ekranda ortaya yerleştirelim
        text = target_emotion.capitalize()
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 4, 3)
        x = (frame.shape[1] - text_width) // 2
        y = 50

        # Metin için bir arka plan (şeffaf dikdörtgen)
        background_x1 = x - 10
        background_y1 = y - text_height - 10
        background_x2 = x + text_width + 10
        background_y2 = y + 10

        # Şeffaf arka plan rengini belirleyelim (örneğin siyah ve %50 şeffaf)
        overlay = frame.copy()
        cv2.rectangle(overlay, (background_x1, background_y1), (background_x2, background_y2), (0, 0, 0), -1)
        alpha = 0.5  # Şeffaflık oranı
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Duygu rengini al
        color = self.emotion_color_map.get(target_emotion, self.color_yellow)

        # Metni ekle
        self.put_text(frame, text, (x, y), color, font_scale=2.0, font=cv2.FONT_HERSHEY_COMPLEX)

    def reset_score_bg(self):
        self.scor_bg_image = self.original_scor_bg_image.copy()

    def show_countdown(self):
        for i in range(self.countdown_duration, 0, -1):
            t = time.time()
            while time.time() - t < 1:
                frame = self.decrease_resolution(self.frame)
                self.put_text(self.scor_bg_image, f"Round starts in {i}", (50, 250), self.color_white)
                self.display_frame(frame, wait_key=1)

    def decrease_resolution(self, frame):
        return cv2.resize(frame, None, fx=self.resolution_scale, fy=self.resolution_scale)

    def draw_transparent_rectangle(self, frame, start_point, end_point, color, opacity):
        overlay = frame.copy()
        cv2.rectangle(overlay, start_point, end_point, color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    def calculate_scores(self, player_scores, emotion_counts):
        return [player_scores[i] / emotion_counts[i] if emotion_counts[i] > 0 else 0 for i in range(len(player_scores))]

    def display_winner(self, winner_text):
        display_winner_start_time = time.time()
        while time.time() - display_winner_start_time < 3:
            self.put_text(self.scor_bg_image, winner_text, (50, 350), (0, 255, 0))
            self.display_frame(self.frame, wait=False)

    def __start_video_capture(self):
        try:
            fps = 25
            wait_sec = (1 / fps) * 0.9
            retry_count = -1
            image_send_count = 0
            while self._is_disposed == False and retry_count < 3:
                if self.__capture == None or self.__capture.isOpened() == False:
                    retry_count += 1
                    self.__capture = cv2.VideoCapture(1)
                    # empty reads
                    for _ in range(10):
                        time.sleep(0.01)
                        _, _ = self.__capture.read()
                ret, frame = self.__capture.read()
                if not ret:
                    retry_count += 1
                    continue
                retry_count = 0

                try:
                    self.frame_read_lock.acquire(blocking=True)
                    self.frame = frame
                finally:
                    image_send_count += 1
                    self.frame_read_lock.release()
                time.sleep(wait_sec)
        finally:
            self._is_disposed = True
            if self.__capture != None:
                self.__capture.release()

    def start_game(self):
        t = threading.Thread(target=self.__start_video_capture, daemon=True)
        t.start()

        time.sleep(5)

        for round_num in range(self.total_rounds):
            target_emotion = random.choice(self.emotions)

            detected_players = False
            player_scores = {}
            player_id_map = {}
            while not detected_players:
                frame = self.decrease_resolution(self.frame)
                faces, labels, ids = self.detect_faces(frame)
                if len(faces) >= 1:
                    detected_players = True
                    for idx, player_id in enumerate(ids):
                        if player_id not in player_id_map:
                            player_id_map[player_id] = idx
                            self.player_colors[player_id] = self.assign_random_color()
                            if player_id not in self.overall_player_scores and player_id is not None:
                                self.overall_player_scores[player_id] = 0
                else:
                    self.put_text(self.scor_bg_image, "Please bring more players!",
                                  (50, 100), (0, 0, 255))
                    self.display_frame(frame, wait=False)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.show_countdown()

            countdown_start_time = time.time()
            player_scores = {player_id: 0 for player_id in player_id_map.keys()}
            emotion_counts = {player_id: 0 for player_id in player_id_map.keys()}

            while time.time() - countdown_start_time < self.round_duration:
                self.put_text(self.scor_bg_image, f"Round {round_num + 1}", (50, 150), (0, 0, 0))

                frame = self.decrease_resolution(self.frame)
                self.display_target_emotion(frame, target_emotion)
                faces, labels, ids = self.detect_faces(frame)

                for i, (face, (x1, y1, x2, y2), player_id) in enumerate(zip(faces, labels, ids)):
                    if player_id in player_id_map:
                        emotion, emotion_scores = self.detect_emotions(face)
                        color = self.player_colors[player_id]
                        self.draw_transparent_rectangle(frame, (x1, y1), (x2, y2), color, opacity=0.5)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        if emotion_scores:
                            score = emotion_scores.get(target_emotion, 0)
                            player_scores[player_id] += score
                            emotion_counts[player_id] += 1
                            self.put_text(frame, f"Player {player_id} - Score: {score:.2f}",
                                          (x1, y1 - 10), (255, 217, 92), font_scale=0.5)

                self.display_frame(frame, wait=False)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            player_avg_scores = self.calculate_scores(list(player_scores.values()), list(emotion_counts.values()))
            for player_id, avg_score in zip(player_scores.keys(), player_avg_scores):
                print(f"Player {player_id} - Average Score: {avg_score:.2f}")
                self.overall_player_scores[player_id] += avg_score  # Kümülatif skoru güncelle

            sorted_players = sorted(self.overall_player_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_players) > 0:
                winner_text = f"Round {round_num + 1} winner: Player {sorted_players[0][0]}"
                self.display_winner(winner_text)

        time.sleep(5)
        cv2.destroyAllWindows()

        # Final score display
        print("Game Over!")
        print("Final Scores:")
        for player_id, score in sorted(self.overall_player_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"Player {player_id}: {score:.2f}")


# Create an instance and start the game
game = EmotionMimicGame()
game.start_game()
