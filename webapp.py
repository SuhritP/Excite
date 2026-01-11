"""Simple ADHD Detection Web App"""
import os
import csv
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

state = {'status': 'ready', 'progress': 0, 'message': 'Ready', 'result': None}

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head><title>ADHD Detection</title>
<style>
body{font-family:Arial;max-width:900px;margin:40px auto;padding:20px;background:#f0f0f0}
h1{color:#333}
.box{background:white;padding:30px;border-radius:10px;margin:20px 0}
button{background:#4CAF50;color:white;padding:15px 30px;border:none;cursor:pointer;font-size:16px;border-radius:5px}
button:hover{background:#45a049}
#result{font-size:32px;font-weight:bold;margin:20px 0}
.adhd{color:#e53935}
.control{color:#43a047}
.progress{background:#ddd;height:20px;border-radius:10px;margin:10px 0}
.bar{background:#4CAF50;height:100%;border-radius:10px;width:0%;transition:0.3s}
video{width:100%;border-radius:10px;background:#000}
</style>
</head>
<body>
<h1>ADHD Detection</h1>
<div class="box">
<h3>Upload Eye Tracking Video</h3>
<input type="file" id="file" accept="video/*"><br><br>
<button onclick="upload()">Analyze Video</button>
</div>
<div class="box" id="vidbox" style="display:none">
<h3>Eye Tracking Preview</h3>
<img id="trackedGif" style="width:100%;border-radius:10px">
</div>
<div class="box">
<h3>Status</h3>
<p id="msg">Ready to analyze</p>
<div class="progress"><div class="bar" id="bar"></div></div>
</div>
<div class="box" id="resbox" style="display:none">
<h3>Result</h3>
<div id="result"></div>
<p>Probability: <span id="prob"></span>%</p>
<p>Confidence: <span id="conf"></span>%</p>
<p>Frames: <span id="frames"></span></p>
</div>
<script>
function upload(){
  var f=document.getElementById('file').files[0];
  if(!f){alert('Select a video');return;}
  var fd=new FormData();fd.append('video',f);
  document.getElementById('msg').textContent='Uploading...';
  document.getElementById('resbox').style.display='none';
  fetch('/upload',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    if(d.success)poll();else document.getElementById('msg').textContent='Error: '+d.error;
  });
}
function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('msg').textContent=d.message;
    document.getElementById('bar').style.width=d.progress+'%';
    if(d.status=='done'){
      document.getElementById('resbox').style.display='block';
      document.getElementById('vidbox').style.display='block';
      document.getElementById('trackedGif').src='/gif?t='+Date.now();
      var r=d.result;
      document.getElementById('result').textContent=r.prediction;
      document.getElementById('result').className=r.prediction=='ADHD'?'adhd':'control';
      document.getElementById('prob').textContent=r.probability;
      document.getElementById('conf').textContent=r.confidence;
      document.getElementById('frames').textContent=r.frames;
    }else if(d.status=='error'){
      document.getElementById('msg').textContent='Error: '+d.message;
    }else{setTimeout(poll,500);}
  });
}
</script>
</body></html>'''

@app.route('/upload', methods=['POST'])
def upload():
    global state
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video'})
    f = request.files['video']
    path = os.path.join('uploads', secure_filename(f.filename))
    f.save(path)
    state = {'status': 'processing', 'progress': 0, 'message': 'Starting...', 'result': None}
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
        
        # Save tracked frames for GIF
        tracked_frames = []
        results = []
        frame_id = 0
        
        state['message'] = 'Processing frames...'
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            state['progress'] = int(frame_id / total * 80)
            state['message'] = f'Tracking eye in frame {frame_id}/{total}'
            
            try:
                # Use teammate's CV model for proper pupil detection
                result = process_frame(frame.copy())
                
                # Resize frame for display
                display_frame = crop_to_aspect_ratio(frame, 320, 240)
                
                if result is not None:
                    # result is ((cx, cy), (w, h), angle) rotated rect
                    center = result[0] if result[0] else (160, 120)
                    size = result[1] if result[1] else (30, 30)
                    
                    px, py = int(center[0] * 320/640), int(center[1] * 240/480)
                    pupil_d = (size[0] + size[1]) / 2 * 320/640
                    
                    # Draw on display frame
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
            
            # Save every 3rd frame for GIF
            if frame_id % 3 == 0:
                tracked_frames.append(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            
            # Gaze angles
            theta_x = (px - 160) / 10
            theta_y = (py - 120) / 10
            
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
        
        # Save CSV
        state['message'] = 'Running ML prediction...'
        state['progress'] = 85
        
        csv_path = 'outputs/metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_id', 'x', 'y', 'dP', 'theta_x', 'theta_y', 'val'])
            w.writerows(results)
        
        # Run ML
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
        
        state['status'] = 'done'
        state['message'] = 'Complete!'
        
    except Exception as e:
        state['status'] = 'error'
        state['message'] = str(e)

if __name__ == '__main__':
    print('='*50)
    print('ADHD Detection App')
    print('Open: http://localhost:5000')
    print('='*50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
