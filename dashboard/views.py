import cv2
import os
import csv
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse, FileResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from ultralytics import YOLO

# Global variable to store live traffic counts for the chart
live_counts = {'Cars': 0, 'Buses': 0, 'Trucks': 0, 'Motorbikes': 0, 'Pedestrians': 0}

def index(request):
    context = {'video_uploaded': False}
    
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        selected_classes = request.POST.getlist('classes')
        
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        uploaded_file_url = fs.path(filename)
        
        request.session['video_path'] = uploaded_file_url
        request.session['selected_classes'] = selected_classes
        
        context['video_uploaded'] = True
        
    return render(request, 'dashboard/index.html', context)

def generate_frames(video_path, selected_classes):
    global live_counts
    # Reset counts when a new stream starts
    live_counts = {'Cars': 0, 'Buses': 0, 'Trucks': 0, 'Motorbikes': 0, 'Pedestrians': 0}
    tracked_unique_ids = set() # Keep track of which IDs we've already counted
    
    model = YOLO('yolov8n.pt') 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    class_map = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7, 'person': 0, 'bicycle': 1}
    class_ids = [class_map[c] for c in selected_classes if c in class_map]
    
    # 1. Setup the Log File
    log_path = os.path.join(settings.MEDIA_ROOT, 'traffic_logs.csv')
    heatmap_layer = None
    frame_count = 0
    
    with open(log_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        # Write the required headers
        log_writer.writerow(['timestamp', 'frame', 'track_id', 'class', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            timestamp = round(frame_count / fps, 2)
            
            # Initialize heatmap blank canvas
            if heatmap_layer is None:
                heatmap_layer = np.zeros_like(frame, dtype=np.uint8)
                
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=class_ids)
            object_present = False
            
            if results[0].boxes.id is not None:
                object_present = True
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    class_name = model.names[cls_id]
                    
                    # --- NEW COUNTING LOGIC FOR CHART ---
                    if track_id not in tracked_unique_ids:
                        tracked_unique_ids.add(track_id)
                        if class_name == 'car': live_counts['Cars'] += 1
                        elif class_name == 'bus': live_counts['Buses'] += 1
                        elif class_name == 'truck': live_counts['Trucks'] += 1
                        elif class_name == 'motorcycle': live_counts['Motorbikes'] += 1
                        elif class_name == 'person': live_counts['Pedestrians'] += 1
                    # ------------------------------------

                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # 2. Log the data to CSV
                    log_writer.writerow([timestamp, frame_count, track_id, class_name, x1, y1, x2, y2])
                    
                    # 3. Add to Heatmap (Draw a glowing circle at the center of the bounding box)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(heatmap_layer, (center_x, center_y), 20, (0, 0, 255), -1)
                    
                    # Draw bounding boxes and IDs
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id} {class_name}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Apply the heatmap over the frame with some transparency
            frame = cv2.addWeighted(frame, 0.8, heatmap_layer, 0.4, 0)
                                
            if not object_present:
                cv2.putText(frame, "NO SELECTED OBJECTS PRESENT", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    cap.release()

def video_feed(request):
    video_path = request.session.get('video_path')
    selected_classes = request.session.get('selected_classes', ['car', 'bus', 'truck'])
    
    if video_path:
        return StreamingHttpResponse(generate_frames(video_path, selected_classes),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    return StreamingHttpResponse("No video uploaded")

# View to let users download the log file
def download_logs(request):
    log_path = os.path.join(settings.MEDIA_ROOT, 'traffic_logs.csv')
    if os.path.exists(log_path):
        return FileResponse(open(log_path, 'rb'), as_attachment=True, filename='traffic_logs.csv')
    return StreamingHttpResponse("Logs not generated yet. Please run a video first.")

# View to serve live chart data to JavaScript
def get_chart_data(request):
    return JsonResponse(live_counts)