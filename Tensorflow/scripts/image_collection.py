import cv2
import os

# List of labels for image categories
labels = ["Hello", "Thanks", "Yes", "No", "ILoveYou"]

# Number of images to capture for each label
number_images = 30

# Directory to save captured images
save_directory = "captured_images"

def capture_images(label):
    # Create a directory for the label if it doesn't exist
    label_directory = os.path.join(save_directory, label)
    os.makedirs(label_directory, exist_ok=True)

    cap = cv2.VideoCapture(0)
    for i in range(number_images):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow('frame', frame)

        # Save image with label and index
        image_path = os.path.join(label_directory, f"{label}_{i}.jpg")
        cv2.imwrite(image_path, frame)

        cv2.waitKey(1000)  # Adjust delay as needed
    cap.release()

def main():
    # Create directory to save images if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    for label in labels:
        print(f"Capturing images for {label}")
        capture_images(label)

    cv2.destroyAllWindows()
main()