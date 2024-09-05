from fer import FER

emotion_detector = FER()

def detect_emotion_fer(face_image):
    result = emotion_detector.detect_emotions(face_image)
    if result:
        emotion_scores = result[0]['emotions']
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        # multiply all emotion scores by 100
        emotion_scores = {k: v*100 for k, v in emotion_scores.items()}
        return dominant_emotion, emotion_scores
    return None, {}


# print(detect_emotion_fer('img.png'))
