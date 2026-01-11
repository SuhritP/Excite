"""ADHD Detection Web App - Beautiful Frontend"""
import os
import csv
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

state = {'status': 'ready', 'progress': 0, 'message': 'Ready', 'result': None, 'metrics': None}

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADHD Detection | Eye Tracking Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #8892b0;
            font-size: 1.1rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(180deg, #00d9ff, #00ff88);
            border-radius: 2px;
        }
        
        .upload-zone {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-zone:hover {
            border-color: #00d9ff;
            background: rgba(0, 217, 255, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .upload-text {
            color: #8892b0;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            border: none;
            padding: 15px 40px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .progress-container {
            margin-top: 20px;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border-radius: 10px;
            transition: width 0.3s ease;
            width: 0%;
        }
        
        .progress-text {
            color: #8892b0;
            font-size: 0.9rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }
        
        .result-main {
            text-align: center;
            padding: 40px;
        }
        
        .result-badge {
            display: inline-block;
            padding: 20px 60px;
            border-radius: 15px;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .result-badge.adhd {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            box-shadow: 0 10px 30px rgba(238, 90, 36, 0.3);
        }
        
        .result-badge.control {
            background: linear-gradient(135deg, #00ff88, #00d9ff);
            color: #1a1a2e;
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
        }
        
        .probability-ring {
            width: 200px;
            height: 200px;
            margin: 30px auto;
            position: relative;
        }
        
        .probability-ring canvas {
            position: absolute;
            top: 0;
            left: 0;
        }
        
        .probability-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        
        .probability-value {
            font-size: 3rem;
            font-weight: 700;
        }
        
        .probability-label {
            color: #8892b0;
            font-size: 0.9rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .metric-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d9ff;
        }
        
        .metric-label {
            color: #8892b0;
            font-size: 0.85rem;
            margin-top: 5px;
        }
        
        .gif-preview {
            width: 100%;
            border-radius: 15px;
            background: #000;
        }
        
        .chart-container {
            position: relative;
            height: 250px;
            margin-top: 20px;
        }
        
        .hidden {
            display: none !important;
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .insights-list {
            list-style: none;
        }
        
        .insights-list li {
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .insights-list li:last-child {
            border-bottom: none;
        }
        
        .insight-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }
        
        .insight-icon.good { background: rgba(0, 255, 136, 0.2); }
        .insight-icon.warning { background: rgba(255, 193, 7, 0.2); }
        .insight-icon.info { background: rgba(0, 217, 255, 0.2); }
        
        .file-name {
            color: #00ff88;
            font-weight: 500;
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 1.8rem; }
            .grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ADHD Detection System</h1>
            <p class="subtitle">AI-Powered Eye Tracking Analysis</p>
        </header>
        
        <!-- Upload Section -->
        <div class="card" id="uploadCard">
            <div class="card-title">Upload Video</div>
            <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📹</div>
                <p class="upload-text">Drag & drop your eye tracking video here<br>or click to browse</p>
                <input type="file" id="fileInput" class="file-input" accept="video/*">
                <button class="btn" id="uploadBtn">Select Video</button>
                <p class="file-name" id="fileName"></p>
            </div>
            <div class="progress-container hidden" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p class="progress-text" id="progressText">Initializing...</p>
            </div>
        </div>
        
        <!-- Eye Tracking Preview -->
        <div class="card hidden" id="previewCard">
            <div class="card-title">Eye Tracking Preview</div>
            <img class="gif-preview" id="trackedGif" alt="Eye tracking visualization">
        </div>
        
        <!-- Results Section -->
        <div class="hidden" id="resultsSection">
            <div class="grid">
                <!-- Main Result -->
                <div class="card fade-in">
                    <div class="card-title">Analysis Result</div>
                    <div class="result-main">
                        <div class="result-badge" id="resultBadge">--</div>
                        <div class="probability-ring">
                            <canvas id="probChart" width="200" height="200"></canvas>
                            <div class="probability-text">
                                <div class="probability-value" id="probValue">--</div>
                                <div class="probability-label">Probability</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Key Metrics -->
                <div class="card fade-in">
                    <div class="card-title">Key Metrics</div>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-value" id="metricConfidence">--</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value" id="metricFrames">--</div>
                            <div class="metric-label">Frames Analyzed</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value" id="metricSaccades">--</div>
                            <div class="metric-label">Saccade Count</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value" id="metricFixations">--</div>
                            <div class="metric-label">Avg Fixation</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="grid">
                <!-- Gaze Pattern Chart -->
                <div class="card fade-in">
                    <div class="card-title">Gaze Movement Pattern</div>
                    <div class="chart-container">
                        <canvas id="gazeChart"></canvas>
                    </div>
                </div>
                
                <!-- Pupil Size Chart -->
                <div class="card fade-in">
                    <div class="card-title">Pupil Diameter Over Time</div>
                    <div class="chart-container">
                        <canvas id="pupilChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Insights -->
            <div class="card fade-in">
                <div class="card-title">Analysis Insights</div>
                <ul class="insights-list" id="insightsList">
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        let gazeChart, pupilChart, probChart;
        
        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            document.getElementById('fileName').textContent = file.name;
            upload(file);
        }
        
        function resetUI() {
            document.getElementById('previewCard').classList.add('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('trackedGif').src = '';
            document.getElementById('progressFill').style.width = '0%';
            if (gazeChart) gazeChart.destroy();
            if (pupilChart) pupilChart.destroy();
        }
        
        function upload(file) {
            resetUI();
            
            const fd = new FormData();
            fd.append('video', file);
            
            document.getElementById('progressContainer').classList.remove('hidden');
            document.getElementById('progressText').textContent = 'Uploading...';
            
            fetch('/upload', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(d => {
                    if (d.success) poll();
                    else {
                        document.getElementById('progressText').textContent = 'Error: ' + d.error;
                    }
                });
        }
        
        function poll() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('progressText').textContent = d.message;
                    document.getElementById('progressFill').style.width = d.progress + '%';
                    
                    if (d.status === 'done') {
                        showResults(d.result, d.metrics);
                    } else if (d.status === 'error') {
                        document.getElementById('progressText').textContent = 'Error: ' + d.message;
                    } else {
                        setTimeout(poll, 500);
                    }
                });
        }
        
        function showResults(result, metrics) {
            document.getElementById('progressContainer').classList.add('hidden');
            document.getElementById('previewCard').classList.remove('hidden');
            document.getElementById('resultsSection').classList.remove('hidden');
            
            // Load GIF
            document.getElementById('trackedGif').src = '/gif?t=' + Date.now();
            
            // Main result
            const badge = document.getElementById('resultBadge');
            badge.textContent = result.prediction;
            badge.className = 'result-badge ' + (result.prediction === 'ADHD' ? 'adhd' : 'control');
            
            // Probability
            document.getElementById('probValue').textContent = result.probability + '%';
            drawProbabilityRing(result.probability);
            
            // Metrics
            document.getElementById('metricConfidence').textContent = result.confidence + '%';
            document.getElementById('metricFrames').textContent = result.frames;
            document.getElementById('metricSaccades').textContent = metrics ? metrics.saccade_count : '--';
            document.getElementById('metricFixations').textContent = metrics ? metrics.avg_fixation + 'ms' : '--';
            
            // Charts
            if (metrics && metrics.gaze_x) {
                drawGazeChart(metrics.gaze_x, metrics.gaze_y);
                drawPupilChart(metrics.pupil_data);
            }
            
            // Insights
            generateInsights(result, metrics);
        }
        
        function drawProbabilityRing(prob) {
            const canvas = document.getElementById('probChart');
            const ctx = canvas.getContext('2d');
            const centerX = 100, centerY = 100, radius = 80;
            
            ctx.clearRect(0, 0, 200, 200);
            
            // Background ring
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.lineWidth = 12;
            ctx.stroke();
            
            // Progress ring
            const endAngle = (prob / 100) * 2 * Math.PI - Math.PI / 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, -Math.PI / 2, endAngle);
            const gradient = ctx.createLinearGradient(0, 0, 200, 200);
            if (prob > 50) {
                gradient.addColorStop(0, '#ff6b6b');
                gradient.addColorStop(1, '#ee5a24');
            } else {
                gradient.addColorStop(0, '#00ff88');
                gradient.addColorStop(1, '#00d9ff');
            }
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 12;
            ctx.lineCap = 'round';
            ctx.stroke();
        }
        
        function drawGazeChart(gazeX, gazeY) {
            const ctx = document.getElementById('gazeChart').getContext('2d');
            if (gazeChart) gazeChart.destroy();
            
            const labels = gazeX.map((_, i) => i);
            gazeChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Gaze X',
                        data: gazeX,
                        borderColor: '#00d9ff',
                        backgroundColor: 'rgba(0, 217, 255, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }, {
                        label: 'Gaze Y',
                        data: gazeY,
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#8892b0' } } },
                    scales: {
                        x: { display: false },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892b0' } }
                    }
                }
            });
        }
        
        function drawPupilChart(pupilData) {
            const ctx = document.getElementById('pupilChart').getContext('2d');
            if (pupilChart) pupilChart.destroy();
            
            const labels = pupilData.map((_, i) => i);
            pupilChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Pupil Diameter',
                        data: pupilData,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#8892b0' } } },
                    scales: {
                        x: { display: false },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892b0' } }
                    }
                }
            });
        }
        
        function generateInsights(result, metrics) {
            const list = document.getElementById('insightsList');
            list.innerHTML = '';
            
            const insights = [];
            
            if (result.prediction === 'ADHD') {
                insights.push({ icon: 'warning', text: 'Higher gaze variability detected, consistent with ADHD patterns' });
            } else {
                insights.push({ icon: 'good', text: 'Gaze patterns within typical range' });
            }
            
            if (result.confidence > 60) {
                insights.push({ icon: 'good', text: 'High confidence prediction (' + result.confidence + '%)' });
            } else {
                insights.push({ icon: 'info', text: 'Moderate confidence - consider longer video for better accuracy' });
            }
            
            insights.push({ icon: 'info', text: result.frames + ' frames analyzed at 30fps' });
            
            if (metrics) {
                if (metrics.saccade_count > 50) {
                    insights.push({ icon: 'warning', text: 'Above average saccade frequency detected' });
                }
                insights.push({ icon: 'info', text: 'Pupil variability: ' + (metrics.pupil_std || 'N/A') });
            }
            
            insights.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = '<span class="insight-icon ' + item.icon + '">' + 
                    (item.icon === 'good' ? '✓' : item.icon === 'warning' ? '!' : 'i') + 
                    '</span><span>' + item.text + '</span>';
                list.appendChild(li);
            });
        }
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload():
    global state
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video'})
    f = request.files['video']
    path = os.path.join('uploads', secure_filename(f.filename))
    f.save(path)
    state = {'status': 'processing', 'progress': 0, 'message': 'Starting...', 'result': None, 'metrics': None}
    threading.Thread(target=process, args=(path,)).start()
    return jsonify({'success': True})

@app.route('/status')
def status():
    return jsonify(state)

@app.route('/gif')
def serve_gif():
    return send_file('outputs/tracked.gif', mimetype='image/gif')

def process(video_path):
    global state
    import cv2
    from eyetrack import process_frame, crop_to_aspect_ratio
    
    try:
        state['message'] = 'Loading video...'
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        tracked_frames = []
        results = []
        gaze_x_list = []
        gaze_y_list = []
        pupil_list = []
        frame_id = 0
        
        state['message'] = 'Processing frames...'
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            state['progress'] = int(frame_id / total * 80)
            state['message'] = f'Tracking eye in frame {frame_id}/{total}'
            
            try:
                result = process_frame(frame.copy())
                display_frame = crop_to_aspect_ratio(frame, 320, 240)
                
                if result is not None:
                    center = result[0] if result[0] else (160, 120)
                    size = result[1] if result[1] else (30, 30)
                    
                    px, py = int(center[0] * 320/640), int(center[1] * 240/480)
                    pupil_d = (size[0] + size[1]) / 2 * 320/640
                    
                    cv2.ellipse(display_frame, (px, py), (int(pupil_d/2), int(pupil_d/2)), 
                               0, 0, 360, (0, 255, 0), 2)
                    cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)
                    valid = 0
                else:
                    px, py, pupil_d = 160, 120, 30
                    valid = 1
                    
            except Exception as e:
                px, py, pupil_d = 160, 120, 30
                valid = 1
                display_frame = cv2.resize(frame, (320, 240))
            
            cv2.putText(display_frame, f'Frame: {frame_id}', (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if frame_id % 3 == 0:
                tracked_frames.append(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            
            theta_x = (px - 160) / 10
            theta_y = (py - 120) / 10
            
            gaze_x_list.append(theta_x)
            gaze_y_list.append(theta_y)
            pupil_list.append(pupil_d)
            
            results.append([frame_id, px, py, pupil_d, theta_x, theta_y, valid])
            frame_id += 1
        
        cap.release()
        
        # Save as GIF
        state['message'] = 'Creating preview GIF...'
        from PIL import Image
        if tracked_frames:
            imgs = [Image.fromarray(f) for f in tracked_frames[:100]]
            imgs[0].save('outputs/tracked.gif', save_all=True, append_images=imgs[1:], 
                        duration=100, loop=0)
        
        # Calculate metrics
        state['message'] = 'Calculating metrics...'
        state['progress'] = 85
        
        # Saccade detection (velocity threshold)
        velocities = []
        for i in range(1, len(gaze_x_list)):
            dx = gaze_x_list[i] - gaze_x_list[i-1]
            dy = gaze_y_list[i] - gaze_y_list[i-1]
            v = np.sqrt(dx**2 + dy**2) * fps
            velocities.append(v)
        
        saccade_threshold = 30  # degrees per second
        saccade_count = sum(1 for v in velocities if v > saccade_threshold)
        
        # Average fixation duration
        fixation_frames = sum(1 for v in velocities if v < saccade_threshold)
        avg_fixation = int((fixation_frames / max(len(velocities), 1)) * (1000 / fps))
        
        # Pupil stats
        pupil_std = round(np.std(pupil_list), 2) if pupil_list else 0
        
        # Subsample for charts (max 200 points)
        step = max(1, len(gaze_x_list) // 200)
        
        metrics = {
            'saccade_count': saccade_count,
            'avg_fixation': avg_fixation,
            'pupil_std': pupil_std,
            'gaze_x': gaze_x_list[::step],
            'gaze_y': gaze_y_list[::step],
            'pupil_data': pupil_list[::step]
        }
        
        # Run ML prediction
        state['message'] = 'Running ML prediction...'
        state['progress'] = 95
        
        from inference import ADHDDetector
        detector = ADHDDetector()
        
        preds = []
        for r in results:
            res = detector.add_frame(r[4], r[5], r[3], r[6])
            if res:
                preds.append(res['probability'])
        
        state['progress'] = 100
        
        if preds:
            avg = np.mean(preds)
            state['result'] = {
                'prediction': 'ADHD' if avg > 0.5 else 'Control',
                'probability': round(avg * 100, 1),
                'confidence': round(abs(avg - 0.5) * 200, 1),
                'frames': len(results)
            }
        else:
            state['result'] = {
                'prediction': 'Need more data',
                'probability': 0,
                'confidence': 0,
                'frames': len(results)
            }
        
        state['metrics'] = metrics
        state['status'] = 'done'
        state['message'] = 'Analysis complete!'
        
    except Exception as e:
        state['status'] = 'error'
        state['message'] = str(e)

if __name__ == '__main__':
    print('='*60)
    print('   ADHD Detection System - Eye Tracking Analysis')
    print('='*60)
    print('   Open: http://localhost:5000')
    print('='*60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
