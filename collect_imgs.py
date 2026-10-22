
import os
import cv2

# Settings
DATA_DIR = "./data"
number_of_classes = 3
dataset_size = 100

# Open the webcam
# Try camera 0 first
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check whether the camera opened successfully
if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    print("Try changing camera index 0 to 1.")
    print("Also check Windows Settings > Privacy & security > Camera.")
    raise SystemExit

# Create the main data folder
os.makedirs(DATA_DIR, exist_ok=True)

# Sign names for the three classes
labels = {
    0: "A",
    1: "B",
    2: "L"
}

# Collect images for each sign
for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    sign_name = labels[j]

    print(f"\nCollecting images for sign {sign_name}")
    print('Show the sign, then press "Q" to start collecting.')

    # Wait until Q is pressed
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("ERROR: Failed to read from webcam.")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        cv2.putText(
            frame,
            f"Ready for sign {sign_name}? Press Q",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collect Images", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Capture images
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("ERROR: Webcam frame unavailable.")
            break

        cv2.putText(
            frame,
            f"Sign: {sign_name} | Image: {counter + 1}/{dataset_size}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collect Images", frame)

        # Save the image
        image_path = os.path.join(class_dir, f"{counter}.jpg")
        cv2.imwrite(image_path, frame)

        counter += 1
        cv2.waitKey(1)

    print(f"Saved {counter} images for sign {sign_name}.")

    if counter < dataset_size:
        print("Image collection stopped because the webcam failed.")
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

print("\nImage collection finished!")
