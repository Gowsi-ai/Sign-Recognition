
import cv2
import mediapipe as mp
import pickle
import os
from collections import Counter

DATA_FILE = "phrases_data.pickle"
SEQUENCE_LENGTH = 40
SAMPLES_PER_PHRASE = 25

PHRASES = [
    "Thank you",
    "Hello",
    "Sorry",
    "How are you?",
    "Please",
    "Good morning"
]

# Load existing recordings so they are not overwritten
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        dataset = pickle.load(f)
    sequences = dataset["sequences"]
    labels = dataset["labels"]
else:
    sequences = []
    labels = []

counts = Counter(labels)
print("Existing samples:", dict(counts))

mp_hands = mp.solutions.hands


def extract_features(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    features = [wrist.x, wrist.y, wrist.z]

    for point in landmarks:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])

    return features


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

stop_collection = False

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for phrase in PHRASES:
            sample_number = counts[phrase]

            if sample_number >= SAMPLES_PER_PHRASE:
                print(f"Skipping {phrase}: already have {sample_number}")
                continue

            print(f"\nCollecting: {phrase}")

            while sample_number < SAMPLES_PER_PHRASE:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Cannot read webcam.")
                    stop_collection = True
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                cv2.putText(
                    frame, f"Phrase: {phrase}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Example: {sample_number + 1}/{SAMPLES_PER_PHRASE}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "S: record | Q: save and quit",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2
                )

                cv2.imshow("Phrase Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    stop_collection = True
                    break

                if key == ord("s"):
                    sequence = []
                    detected_frames = 0

                    for _ in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb)

                        if results.multi_hand_landmarks:
                            features = extract_features(
                                results.multi_hand_landmarks[0]
                            )
                            detected_frames += 1
                        else:
                            features = [0.0] * 66

                        sequence.append(features)

                        cv2.putText(
                            frame, f"Recording: {phrase}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2
                        )
                        cv2.imshow("Phrase Collection", frame)
                        cv2.waitKey(1)

                    if len(sequence) != SEQUENCE_LENGTH:
                        print("Recording failed. Try again.")
                        continue

                    if detected_frames < 30:
                        print("Hand lost too often. Repeat the sign.")
                        continue

                    sequences.append(sequence)
                    labels.append(phrase)
                    sample_number += 1
                    counts[phrase] += 1

                    # Save after every successful sample
                    with open(DATA_FILE, "wb") as f:
                        pickle.dump(
                            {"sequences": sequences, "labels": labels},
                            f
                        )

                    print(
                        f"Saved {phrase}: "
                        f"{sample_number}/{SAMPLES_PER_PHRASE}"
                    )

                if stop_collection:
                    break

            if stop_collection:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\nFinal sample counts:", dict(Counter(labels)))
print("Dataset saved to", DATA_FILE)
