import requests
import os

def test_cloud_upload_and_process():
    # Test file
    test_file = 'static/images/shazwithhotdog.png'
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    # Get upload URL
    response = requests.post(
        'http://localhost:8080/api/upload-url',
        json={
            'fileName': 'shazwithhotdog.png',
            'contentType': 'image/png'
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to get upload URL: {response.text}")
        return
        
    upload_data = response.json()
    upload_url = upload_data['url']
    file_name = upload_data['fileName']
    
    print(f"Got upload URL: {upload_url}")
    
    # Upload file
    with open(test_file, 'rb') as f:
        response = requests.put(
            upload_url,
            data=f,
            headers={'Content-Type': 'image/png'}
        )
        
    if response.status_code != 200:
        print(f"Failed to upload file: {response.text}")
        return
        
    print(f"Successfully uploaded file to {file_name}")
    
    # Process the uploaded file
    response = requests.post(
        'http://localhost:8080/api/process_from_storage',
        json={
            'fileUrls': [f'gs://shaz-video-editor-bucket/{file_name}'],
            'style': 'Fast & Energized'
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to process file: {response.text}")
        return
        
    result = response.json()
    print(f"Processing successful: {result}")
    
if __name__ == "__main__":
    test_cloud_upload_and_process() 