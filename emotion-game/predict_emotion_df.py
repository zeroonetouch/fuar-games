from deepface import DeepFace


def detect_emotion_df(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        if analysis:
            emotion_scores = analysis[0]['emotion']
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            return dominant_emotion, emotion_scores
    except:
        return None


# print(detect_emotion_df('img.png'))
