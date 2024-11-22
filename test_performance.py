import time
import requests
import os

def test_video_processing(url, video_path, style="Fast & Energized"):
    start_time = time.time()
    
    # Prepare the files
    files = [('files', open(video_path, 'rb'))]
    data = {'style': style}
    
    # Make the request
    response = requests.post(f"{url}/api/process_video", files=files, data=data)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return processing_time

if __name__ == "__main__":
    # Test local Docker
    test_video_processing("http://localhost:8080", "path/to/test/video.mp4")
    
    # Test GCP
    test_video_processing("https://video-editor-boplga2fbq-uc.a.run.app", "path/to/test/video.mp4") 