from flask import Flask, request, jsonify, send_from_directory, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import time
import shutil
from transcriber import (
    process_multiple_from_gcs,
    process_videos_cloud,
    time_to_seconds,
    main
)
import subprocess
import traceback
from flask_cors import CORS
import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging
import psutil
import concurrent.futures
import uuid
import sys
import tempfile
import json
from editing_presets import VideoStyle, STYLE_DEFINITIONS
from collections import defaultdict
import threading
from progress_tracker import (
    update_progress,
    get_progress,
    get_stage_details,
    get_current_stage
)
from typing import List, Tuple
from google.cloud import speech_v1
from google.cloud.speech_v1 import types
from editing_instructions import EditingInstructionsGenerator
from dotenv import load_dotenv
from google.cloud import storage
import datetime

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# At the top of app.py, before creating the Flask app
def init_credentials():
    """Initialize GCS credentials"""
    try:
        # Try both possible credential files
        cred_files = [
            'video-editor-credentials.json',
            'video-editor-434002-b3492f400f55.json'
        ]
        
        for cred_file in cred_files:
            if os.path.exists(cred_file):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(cred_file)
                return True
                
        raise FileNotFoundError("No credentials file found")
        
    except Exception as e:
        print(f"Error initializing credentials: {str(e)}")
        return False

# Initialize credentials before creating app
if not init_credentials():
    raise RuntimeError("Failed to initialize GCS credentials")

app = Flask(__name__, 
    static_folder='static',
    static_url_path=''
)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

# Constants
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'quicktime'}
UPLOAD_FOLDER = 'uploads'
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'

# Create necessary folders
for folder in [UPLOAD_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['GCS_BUCKET_NAME'] = 'shaz-video-editor-bucket'
app.config['MAX_CONTENT_LENGTH'] = None  # Remove size limit
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this route specifically for the root path
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Keep your existing route for other paths
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    files = []
    file_paths = []
    for key, file in request.files.items():
        if key.startswith('file'):
            if file and allowed_file(file.filename):
                files.append(file)
    
    for key, value in request.form.items():
        if key.startswith('file'):
            if value.startswith('http'):  # Dropbox link
                response = requests.get(value)
                file = io.BytesIO(response.content)
                filename = f"dropbox_file_{key}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(file.getvalue())
                files.append(filepath)
            elif value.startswith('drive'):  # Google Drive ID
                file_id = value.split(':')[1]
                creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
                drive_service = build('drive', 'v3', credentials=creds)
                request = drive_service.files().get_media(fileId=file_id)
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                file.seek(0)
                filename = f"gdrive_file_{key}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(file.getvalue())
                files.append(filepath)
    
    style = request.form.get('style', 'Fast & Energized')
    
    app.logger.info(f"Received {len(files)} files with style: {style}")
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(save_file, file): file for file in files}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future.result()
                if file_path:
                    file_paths.append(file_path)
        
        app.logger.info("Calling main function")
        result, transcripts = main(file_paths, style)
        app.logger.info(f"Main function result: {result}")
        
        if result:
            processed_filename = f"processed_video_{int(time.time())}.mp4"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            shutil.move(result, processed_filepath)
            
            download_url = url_for('download_file', filename=processed_filename, _external=True)
            
            return jsonify({
                'success': True,
                'download_url': download_url,
                'transcripts': transcripts
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Video processing failed'
            }), 400
    except Exception as e:
        app.logger.error(f"Error in transcribe function: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        cleanup_files(file_paths)

def save_file(file):
    try:
        if not isinstance(file, str):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            filepath = file
        logger.info(f"Saved file: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

def cleanup_files(file_paths):
    for filepath in file_paths:
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up file {filepath}: {str(e)}")

@app.route('/api/large_file_upload', methods=['POST'])
def large_file_upload():
    bucket_name = app.config['GCS_BUCKET_NAME']
    blob_name = f"uploads/{uuid.uuid4()}.mp4"
    signed_url = generate_signed_url(bucket_name, blob_name)
    
    return jsonify({"upload_url": signed_url, "blob_name": blob_name})

@app.route('/api/process_large_file', methods=['POST'])
def process_large_file():
    blob_name = request.json['blob_name']
    style = request.json.get('style', 'Fast & Energized')
    
    result, transcripts = process_from_gcs(blob_name, style)
    
    return jsonify({
        'success': True,
        'download_url': result['output_url'],
        'transcripts': transcripts
    }), 200

@app.route('/api/process_video', methods=['POST'])
def process_video_route():
    task_id = str(uuid.uuid4())
    file_paths = []
    
    try:
        # Initial progress
        update_progress(task_id, 'upload', 0, 'Starting upload')
        
        # Get files from request
        files = request.files.getlist('files')
        if not files:
            raise ValueError('No files provided')
            
        # Get style parameter
        style = request.form.get('style', 'Fast & Energized')
        
        # Save files and get paths
        update_progress(task_id, 'upload', 20, 'Saving uploaded files')
        file_paths = []
        for idx, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                file_paths.append(filepath)
                progress = 20 + (40 * (idx + 1) // len(files))
                update_progress(task_id, 'upload', progress, f'Saved file {idx + 1} of {len(files)}')
                
        if not file_paths:
            raise ValueError('No valid files uploaded')
            
        # Process videos with task_id
        update_progress(task_id, 'process', 60, 'Processing videos')
        result, transcripts = main(file_paths, style, task_id)
        
        if not result:
            raise ValueError("Video processing failed")
            
        update_progress(task_id, 'finalize', 100, 'Complete')
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        update_progress(task_id, 'error', 0, f'Error: {str(e)}')
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    finally:
        # Clean up uploaded files
        if file_paths:
            cleanup_files(file_paths)

@app.route('/downloads/<filename>')
def download_file(filename):
    app.logger.info(f"Download requested for file: {filename}")
    file_path = os.path.join('/tmp', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        app.logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/progress')
def progress():
    def generate():
        for i in range(0, 101, 5):
            time.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'status': 'Processing', 'progress': i})}\n\n"
        yield f"data: {json.dumps({'status': 'Complete', 'progress': 100})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

def generate_signed_url(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="PUT",
        content_type="application/octet-stream",
    )
    return url

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for all errors"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'details': str(e)
    }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'details': str(error)
    }), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    """Handle 405 errors"""
    logger.warning(f"Method not allowed: {request.method} {request.url}")
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed',
        'allowed_methods': error.valid_methods
    }), 405

@app.route('/health')
def health_check():
    try:
        # Check directories
        directories_status = {}
        for folder in [UPLOAD_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER]:
            directories_status[folder] = os.path.exists(folder) and os.access(folder, os.W_OK)
        
        # Check credentials
        creds_status = os.path.exists('video-editor-credentials.json')
        
        return jsonify({
            'status': 'healthy',
            'directories': directories_status,
            'credentials': creds_status,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'disk_usage': psutil.disk_usage('/').percent
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    try:
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Serve the file from the output directory
        return send_from_directory('output', filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error serving output file: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/styles', methods=['GET'])
def get_styles():
    return jsonify({
        style.value: STYLE_DEFINITIONS[style]
        for style in VideoStyle
    })

@app.route('/api/process-video', methods=['POST'])
def process_video():
    try:
        video_file = request.files.get('video')
        style_name = request.form.get('style', 'professional')
        
        # Validate style
        try:
            style = VideoStyle(style_name)
        except ValueError:
            style = VideoStyle.PROFESSIONAL

        # Process video with style
        result = transcriber.transcribe_and_analyze(
            video_file.filename,
            style=style
        )

        # Apply editing instructions
        # ... (implement video processing based on instructions)

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    if request.method == 'OPTIONS':
        return response
    return response

# Add this new route for detailed progress
@app.route('/api/process_status/<task_id>', methods=['GET'])
def get_process_status(task_id):
    try:
        current_stage = get_current_stage(task_id)
        progress = get_progress(task_id)
        details = get_stage_details(task_id)
        
        logger.info(f"Status request for task {task_id}: stage={current_stage}, progress={progress}, details={details}")
        
        response = {
            'status': 'processing',
            'current_stage': current_stage,
            'progress': progress,
            'details': details
        }
        
        logger.info(f"Sending response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting process status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-url', methods=['POST'])
def get_upload_url():
    """Generate signed URL for GCS upload"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            raise VideoEditorException('No JSON data received')
            
        file_name = data.get('fileName')
        if not file_name:
            raise VideoEditorException('fileName is required')
            
        # Generate blob name with timestamp
        timestamp = int(time.time())
        blob_name = f"uploads/{timestamp}_{file_name}"
        
        # Get GCS client and bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket('shaz-video-editor-bucket')
        blob = bucket.blob(blob_name)
        
        # Generate signed URL with correct headers
        url = blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=15),
            method='PUT',
            content_type='video/quicktime',
            headers={
                'Content-Type': 'video/quicktime',
                'x-goog-content-length-range': '0,5368709120'  # 5GB limit
            }
        )
        
        return jsonify({
            'status': 'success',
            'url': url,
            'fileName': blob_name
        })
            
    except storage.exceptions.GoogleCloudError as e:
        logger.error(f"Storage error: {str(e)}")
        raise StorageError(f"Storage service error: {str(e)}")
        
    except VideoEditorException as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/upload/<path:filename>', methods=['PUT'])
def handle_direct_upload(filename):
    try:
        if not request.data:
            return jsonify({'error': 'No file data received'}), 400
            
        storage_client = storage.Client()
        bucket = storage_client.bucket(app.config['GCS_BUCKET_NAME'])
        blob = bucket.blob(f'uploads/{filename}')
        
        content_type = request.headers.get('Content-Type', 'application/octet-stream')
        blob.upload_from_string(
            request.data,
            content_type=content_type
        )
        
        return jsonify({
            'status': 'success',
            'path': f'uploads/{filename}'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_from_storage', methods=['POST'])
def process_from_storage():
    """Process videos from GCS storage"""
    try:
        data = request.get_json()
        file_urls = data.get('fileUrls', [])
        style = data.get('style', 'Fast & Energized')
        task_id = str(uuid.uuid4())
        
        update_progress(task_id, 'initialize', 0, 'Starting cloud processing')
        
        # Process files from GCS
        result_url, transcripts = process_multiple_from_gcs(file_urls, style, task_id)
        
        if result_url:
            return jsonify({
                'status': 'success',
                'task_id': task_id,
                'download_url': result_url,
                'transcripts': transcripts
            })
        
        return jsonify({
            'status': 'error',
            'message': 'Failed to process videos'
        }), 500
        
    except Exception as e:
        logger.error(f"Error processing from storage: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Add helper function for signed URLs
def generate_download_url(bucket_name: str, blob_name: str, file_format: str) -> str:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    content_type = {
        '.mov': 'video/quicktime',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo'
    }.get(file_format, 'video/mp4')
    
    # Use integer seconds for expiration
    url = blob.generate_signed_url(
        version='v4',
        expiration=3600,  # 1 hour in seconds
        method='GET',
        response_type=content_type,
        query_parameters={
            'response-content-type': content_type,
            'response-content-disposition': f'attachment; filename="edited_video{file_format}"'
        }
    )
    return url

@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error occurred',
        'error': str(e)
    }), 500

@app.route('/upload', methods=['PUT'])
def handle_upload():
    try:
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            storage_client = storage.Client()
            bucket = storage_client.bucket(app.config['GCS_BUCKET_NAME'])
            blob = bucket.blob(f'uploads/{filename}')
            blob.upload_from_file(file)
            return jsonify({'status': 'success', 'path': f'uploads/{filename}'})
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def generate_signed_url(blob, method='GET', content_type=None, expiration_minutes=15):
    """Utility function to generate signed URLs consistently"""
    params = {
        'version': 'v4',
        'expiration': datetime.timedelta(minutes=expiration_minutes),
        'method': method
    }
    
    if content_type:
        params['content_type'] = content_type
        
    if method == 'GET':
        params['response_type'] = content_type
        params['query_parameters'] = {
            'response-content-disposition': f'attachment; filename="edited_video.mp4"',
            'response-content-type': content_type
        }
        
    return blob.generate_signed_url(**params)

def check_credentials():
    """Verify GCS credentials are available"""
    try:
        creds_file = 'video-editor-credentials.json'
        if not os.path.exists(creds_file):
            raise FileNotFoundError(f"Credentials file not found: {creds_file}")
            
        # Set environment variable for credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(creds_file)
        
        # Test credentials by creating client
        storage_client = storage.Client()
        bucket = storage_client.bucket('shaz-video-editor-bucket')
        
        return True
        
    except Exception as e:
        logger.error(f"Credentials error: {str(e)}")
        return False

if __name__ == '__main__':
    # Add debug=True and use_reloader=True
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        debug=True,
        use_reloader=True
    )
