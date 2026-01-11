"""
ADHD Detection Web App - Basic Frontend

Upload a video → CV model tracks eyes → ML model predicts ADHD
"""

import os
import cv2
import numpy as np
import pandas as pd
import csv
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Global state for processing
processing_state = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'result': None,
    'video_path': None,
    'output_video': None
}

# Import CV model functions (from teammate's code)
from eyetrack import crop_to_aspect_ratio, get_darkest_area, process_frame

def process_video_with_tracking(video_path):
    """Process video with CV model and return tracking data."""
    global processing_state
    
    processing_state['status'] = 'processing'
    processing_state['message'] = 'Starting eye tracking...'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        processing_state['status'] = 'error'
        processing_state['message'] = 'Could not open video'
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video with tracking overlay
    output_path = 'outputs/tracked_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    results = []
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = int((frame_id / total_frames) * 100) if total_frames > 0 else 0
        processing_state['progress'] = progress
        processing_state['message'] = f'Processing frame {frame_id}/{total_frames}'
        
        # Resize frame
        frame = crop_to_aspect_ratio(frame, 640, 480)
        
        try:
            # Use teammate's process_frame function
            result = process_frame(frame)
            
            if result and result[0] is not None:
                pupil_center, pupil_diameter, glint_positions = result
                px, py = pupil_center if pupil_center else (0, 0)
                
                # Draw tracking on frame
                if pupil_center:
                    cv2.circle(frame, (int(px), int(py)), int(pupil_diameter/2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1)
                
                # Calculate gaze angles
                cx, cy = 320, 240
                theta_x = (px - cx) / 10
                theta_y = (py - cy) / 10
                
                results.append([
                    frame_id, px, py, pupil_diameter or 0,
                    theta_x, theta_y,
                    np.sqrt(theta_x**2 + theta_y**2),
                    0  # Valid
                ])
            else:
                results.append([frame_id, 0, 0, 0, 0, 0, 0, 1])
                
        except Exception as e:
            results.append([frame_id, 0, 0, 0, 0, 0, 0, 1])
        
        # Add status text
        cv2.putText(frame, f'Frame: {frame_id}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_id += 1
    
    cap.release()
    out.release()
    
    # Save metrics
    metrics_path = 'outputs/pupil_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_id', 'circle_x_px', 'circle_y_px', 'pupil_diameter_px',
            'theta_x_deg', 'theta_y_deg', 'theta_radial_deg', 'validity_id'
        ])
        writer.writerows(results)
    
    processing_state['message'] = 'Eye tracking complete'
    processing_state['output_video'] = output_path
    
    return metrics_path


def run_ml_prediction(metrics_path):
    """Run ML model on tracking data."""
    global processing_state
    
    processing_state['message'] = 'Running ADHD prediction...'
    
    from inference import ADHDDetector
    
    # Load tracking data
    df = pd.read_csv(metrics_path)
    
    # Rename columns
    df = df.rename(columns={
        'theta_x_deg': 'x',
        'theta_y_deg': 'y', 
        'pupil_diameter_px': 'dP',
        'validity_id': 'val'
    })
    
    detector = ADHDDetector()
    predictions = []
    
    for i, row in df.iterrows():
        result = detector.add_frame(
            float(row['x']), float(row['y']),
            float(row['dP']), int(row['val'])
        )
        if result:
            predictions.append(result)
    
    if predictions:
        probs = [p['probability'] for p in predictions]
        avg_prob = np.mean(probs)
        
        return {
            'prediction': 'ADHD' if avg_prob > 0.5 else 'Control',
            'probability': round(avg_prob * 100, 1),
            'confidence': round(abs(avg_prob - 0.5) * 200, 1),
            'windows': len(predictions),
            'frames': len(df)
        }
    else:
        return {
            'prediction': 'Insufficient Data',
            'probability': 0,
            'confidence': 0,
            'windows': 0,
            'frames': len(df)
        }


def process_pipeline(video_path):
    """Full pipeline: CV tracking → ML prediction."""
    global processing_state
    
    try:
        # Step 1: Eye tracking
        metrics_path = process_video_with_tracking(video_path)
        
        if metrics_path:
            # Step 2: ML prediction
            result = run_ml_prediction(metrics_path)
            processing_state['result'] = result
            processing_state['status'] = 'complete'
            processing_state['message'] = 'Analysis complete!'
        else:
            processing_state['status'] = 'error'
            processing_state['message'] = 'Eye tracking failed'
            
    except Exception as e:
        processing_state['status'] = 'error'
        processing_state['message'] = str(e)


@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>ADHD Detection</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        button { background: #4CAF50; color: white; padding: 15px 30px; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        #status { margin: 20px 0; padding: 20px; background: #f5f5f5; }
        #result { margin: 20px 0; padding: 20px; background: #e8f5e9; display: none; }
        .progress { width: 100%; background: #ddd; height: 20px; margin: 10px 0; }
        .progress-bar { height: 100%; background: #4CAF50; width: 0%; transition: width 0.3s; }
        video { max-width: 100%; margin: 20px 0; }
        .big-result { font-size: 48px; font-weight: bold; }
        .adhd { color: #f44336; }
        .control { color: #4CAF50; }
    </style>
</head>
<body>
    <h1>ADHD Detection from Eye Tracking</h1>
    
    <div class="upload-box">
        <input type="file" id="videoFile" accept="video/*">
        <br><br>
        <button onclick="uploadVideo()">Upload & Analyze</button>
    </div>
    
    <div id="status">
        <strong>Status:</strong> <span id="statusText">Ready</span>
        <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
    </div>
    
    <div id="videoContainer" style="display:none;">
        <h3>Eye Tracking Preview</h3>
        <video id="trackedVideo" controls></video>
    </div>
    
    <div id="result">
        <h2>Results</h2>
        <div class="big-result" id="prediction"></div>
        <p><strong>Probability:</strong> <span id="probability"></span>%</p>
        <p><strong>Confidence:</strong> <span id="confidence"></span>%</p>
        <p><strong>Frames Analyzed:</strong> <span id="frames"></span></p>
    </div>
    
    <script>
        function uploadVideo() {
            const file = document.getElementById('videoFile').files[0];
            if (!file) { alert('Please select a video'); return; }
            
            const formData = new FormData();
            formData.append('video', file);
            
            document.getElementById('statusText').textContent = 'Uploading...';
            document.getElementById('result').style.display = 'none';
            
            fetch('/upload', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        checkStatus();
                    } else {
                        document.getElementById('statusText').textContent = 'Error: ' + data.error;
                    }
                });
        }
        
        function checkStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('statusText').textContent = data.message;
                    document.getElementById('progressBar').style.width = data.progress + '%';
                    
                    if (data.status === 'complete') {
                        showResult(data.result);
                        if (data.output_video) {
                            document.getElementById('trackedVideo').src = '/video/' + data.output_video;
                            document.getElementById('videoContainer').style.display = 'block';
                        }
                    } else if (data.status === 'error') {
                        document.getElementById('statusText').textContent = 'Error: ' + data.message;
                    } else {
                        setTimeout(checkStatus, 500);
                    }
                });
        }
        
        function showResult(result) {
            document.getElementById('result').style.display = 'block';
            const pred = document.getElementById('prediction');
            pred.textContent = result.prediction;
            pred.className = 'big-result ' + (result.prediction === 'ADHD' ? 'adhd' : 'control');
            document.getElementById('probability').textContent = result.probability;
            document.getElementById('confidence').textContent = result.confidence;
            document.getElementById('frames').textContent = result.frames;
        }
    </script>
</body>
</html>
'''


@app.route('/upload', methods=['POST'])
def upload():
    global processing_state
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Reset state
    processing_state = {
        'status': 'starting',
        'progress': 0,
        'message': 'Starting...',
        'result': None,
        'video_path': filepath,
        'output_video': None
    }
    
    # Start processing in background
    thread = threading.Thread(target=process_pipeline, args=(filepath,))
    thread.start()
    
    return jsonify({'success': True})


@app.route('/status')
def status():
    return jsonify(processing_state)


@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_file(filename)


if __name__ == '__main__':
    print("="*50)
    print("ADHD Detection Web App")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
