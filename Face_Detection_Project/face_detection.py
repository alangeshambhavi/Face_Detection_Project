
import cv2
import os

# --------------------------------
# STEP 1: Load Haar Cascade
# --------------------------------

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("Error: Face Cascade not loaded!")
    exit()

# --------------------------------
# STEP 2: Load Input Image
# --------------------------------

image_path = "image.jpg"

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# --------------------------------
# STEP 3: Convert to Grayscale
# --------------------------------

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# --------------------------------
# STEP 4: Detect Faces
# --------------------------------

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# --------------------------------
# STEP 5: Draw Rectangle
# --------------------------------

for (x, y, w, h) in faces:

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# --------------------------------
# STEP 6: Display Result
# --------------------------------

print("Total Faces Detected:", len(faces))

if len(faces) == 0:
    print("No faces detected.")
else:
    print("Face detection successful!")

cv2.imshow("Face Detection Project", img)

# --------------------------------
# STEP 7: Save Output Image
# --------------------------------

cv2.imwrite("output.jpg", img)

print("Output saved as output.jpg")

# Press any key to close window
cv2.waitKey(0)
cv2.destroyAllWindows()
