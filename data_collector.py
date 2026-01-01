import os
import cv2
import time
import uuid

# Base directory
IMAGE_PATH = "CollectedImages"

# Labels to collect
labels = ['Hello', 'Yes', 'No', 'Thanks', 'IloveYou', 'Please']

# Images per label
NUMBER_OF_IMAGES = 5

# Create base directory
os.makedirs(IMAGE_PATH, exist_ok=True)

for label in labels:
    label_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(" Camera could not be opened")

    print(f"Collecting images for label: {label}")
    print("⏳ Starting in 5 seconds...")
    time.sleep(5)

    for img_num in range(NUMBER_OF_IMAGES):
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to capture frame. Skipping...")
            continue

        image_name = f"{label}.{uuid.uuid4()}.jpg"
        image_path = os.path.join(label_path, image_name)

        cv2.imwrite(image_path, frame)
        cv2.imshow("Image Collection", frame)

        print(f"Saved: {image_path}")
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Manual exit requested")
            break

    cap.release()
    cv2.destroyAllWindows()

print(" Image collection completed successfully.")

