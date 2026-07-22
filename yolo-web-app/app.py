import os
import cv2
from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Setup upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# ----------------- 📸 Image Detection Endpoint -----------------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process image with YOLO
    img = cv2.imread(filepath)
    results = model(img)[0]
    annotated_img = results.plot()

    # Save output image
    output_filename = 'detected_' + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, annotated_img)

    return f"/uploads_file/{output_filename}"

# Serve uploaded / processed files
@app.route('/uploads_file/<filename>')
def serve_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return Response(open(file_path, 'rb').read(), mimetype='image/jpeg')
    return "File not found", 404

# ----------------- 🎥 Video Streaming Generator -----------------
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip frames for fast cloud processing
        if frame_count % 2 != 0:
            continue

        frame_resized = cv2.resize(frame, (480, 360))
        results = model(frame_resized, conf=0.3)[0]
        annotated_frame = results.plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(video_path):
        return "Video not found", 404
        
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video uploaded", 400
        
    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return f"/video_feed/{filename}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
