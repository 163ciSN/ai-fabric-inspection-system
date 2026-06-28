

from flask import Flask, render_template, request
import os
import cv2
import uuid

app = Flask(__name__)

# =====================================================
# FOLDER CONFIGURATION
# =====================================================

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def allowed_file(filename):
    """
    Check whether the uploaded file has an allowed image extension.
    """
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def generate_unique_filename(filename):
    """
    Generate a unique filename to avoid overwriting files.
    Example:
    9d84f2d3_fabric.jpg
    """
    unique_id = uuid.uuid4().hex[:8]
    return f"{unique_id}_{filename}"


def process_image(image_path, output_path):
    """
    Perform AI-based fabric inspection using OpenCV.
    Detect edges and highlight possible defects.
    """

    # Read image
    image = cv2.imread(image_path)

    # Resize image for consistent display
    image = cv2.resize(image, (700, 500))

    processed = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce image noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Detect contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0

    # Draw rectangles around detected regions
    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 150:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                processed,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

            cv2.putText(
                processed,
                "Defect Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            defect_count += 1

    # Draw information panel
    cv2.rectangle(
        processed,
        (10, 10),
        (340, 140),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        processed,
        "AI FABRIC INSPECTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        processed,
        f"Defects Found: {defect_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        processed,
        "Inspection Status: COMPLETE",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Save processed image
    cv2.imwrite(output_path, processed)

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    """
    Display the homepage.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "No file selected."

    image = request.files["image"]

    if image.filename == "":
        return "No file selected."

    if not allowed_file(image.filename):
        return "Only PNG, JPG and JPEG images are allowed."

    # Generate unique filename
    filename = generate_unique_filename(image.filename)

    original_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    processed_filename = "processed_" + filename

    processed_path = os.path.join(
        PROCESSED_FOLDER,
        processed_filename
    )

    # Save uploaded image
    image.save(original_path)

    # Process image
    process_image(
        original_path,
        processed_path
    )

    return render_template(
        "results.html",
        original_image=original_path,
        processed_image=processed_path
    )

# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )