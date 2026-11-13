
import cv2

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Face detector not loaded!")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

print("Webcam Face Detection Started")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display face count
    cv2.putText(
        frame,
        "Faces: " + str(len(faces)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Show webcam
    cv2.imshow("Live Face Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
