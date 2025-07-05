import pathlib
import cv2
import matplotlib.colors as mcolors
from fer import FER
import time


def get_bgr(color_name):
    try:
        rgb = mcolors.to_rgb(color_name)
        return (int(255 * rgb[2]), int(255 * rgb[1]), int(255 * rgb[0]))
    except ValueError:
        return (255, 255, 255)  # fallback


# Base emotion colors
emotion_colors = {
    "happy": get_bgr("yellow"),
    "sad": get_bgr("blue"),
    "angry": get_bgr("red"),
    "disgust": get_bgr("limegreen"),
    "fear": get_bgr("purple"),
    "surprise": get_bgr("orange"),
    "neutral": get_bgr("gray"),
    "unknown": get_bgr("white"),
}

# Add colors for extended feelings
emotion_colors.update({
    "excited": get_bgr("gold"),
    "bored": get_bgr("lightgray"),
    "anxious": get_bgr("indigo"),
    "frustrated": get_bgr("darkred"),
    "melancholy": get_bgr("slateblue"),
    "amazed": get_bgr("deepskyblue"),
})


def interpret_extended_emotions(emotions):
    """
    Interpret raw emotion scores dict and return a more detailed feeling string.
    """
    max_emotion = max(emotions, key=emotions.get)

    # Custom logic for extended feelings
    if max_emotion == "happy" and emotions.get("surprise", 0) > 0.3:
        return "excited"
    if max_emotion == "neutral" and emotions.get("sad", 0) > 0.3:
        return "bored"
    if max_emotion == "fear" and emotions.get("surprise", 0) > 0.4:
        return "anxious"
    if max_emotion == "angry" and emotions.get("disgust", 0) > 0.5:
        return "frustrated"
    if max_emotion == "sad" and emotions.get("neutral", 0) > 0.4:
        return "melancholy"
    if max_emotion == "surprise" and emotions.get("happy", 0) > 0.3:
        return "amazed"

    return max_emotion


def main():
    cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    if face_cascade.empty():
        print("Error loading Haar cascade.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("Expression detection running. Press 'q' to quit.")

    detector = FER(mtcnn=True)

    last_emotions = {}
    emotion_timeout = 2  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for i, (x, y, w, h) in enumerate(faces):
                face_roi = frame[y:y + h, x:x + w]
                face_key = f"{x}_{y}"

                expression = "unknown"
                now = time.time()

                if (face_key not in last_emotions) or (now - last_emotions[face_key]["time"] > emotion_timeout):
                    result = detector.detect_emotions(face_roi)
                    if result:
                        emotions = result[0]["emotions"]
                        expression = interpret_extended_emotions(emotions)
                        last_emotions[face_key] = {
                            "expression": expression,
                            "time": now
                        }
                    else:
                        expression = last_emotions.get(face_key, {}).get("expression", "unknown")
                else:
                    expression = last_emotions[face_key]["expression"]

                rect_color = emotion_colors.get(expression, emotion_colors["unknown"])

                # Draw rectangle and label with color
                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 3)
                cv2.putText(frame, expression.capitalize(), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2)

            cv2.imshow("Expression Detection (Inside Out Style)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()