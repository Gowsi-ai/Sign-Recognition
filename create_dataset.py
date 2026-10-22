
import os
import cv2
import pickle
import mediapipe as mp

DATA_DIR = "./data"

mp_hands = mp.solutions.hands

data = []
labels = []

with mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
) as hands:

    # Visit each class folder: 0, 1, 2
    for dir_ in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, dir_)

        if not os.path.isdir(class_dir):
            continue

        if dir_ not in ["0", "1", "2"]:
            continue

        print(f"Processing class {dir_}...")

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
            )
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                # Use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]

                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                data_aux = []

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

print("Total hand samples detected:", len(data))

if len(data) == 0:
    print("No hands detected. Check your images in data/0, data/1, and data/2.")
else:
    with open("data.pickle", "wb") as f:
        pickle.dump(
            {"data": data, "labels": labels},
            f
        )

    print("Success! data.pickle has been created.")
