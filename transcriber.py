import os
import subprocess
from google.cloud import speech_v1
from google.oauth2 import service_account
from openai import OpenAI
from dotenv import load_dotenv
import time
import tempfile
import json
import re
import logging
from werkzeug.utils import secure_filename
import emoji
from editing_presets import VideoStyle, STYLE_DEFINITIONS, map_frontend_style_to_enum
import random
import traceback
from google.cloud import storage
from editing_instructions import EditingInstructionsGenerator, apply_ai_style
from typing import Tuple, List, Dict
from progress_tracker import update_progress
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import datetime
import shutil
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, AudioFileClip, TextClip, CompositeVideoClip, transfx



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_credentials():
    try:
        credentials_path = 'video-editor-credentials.json'
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
            
        logger.info(f"Loading credentials from: {credentials_path}")
        return service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        raise

speech_client = speech_v1.SpeechClient(credentials=get_credentials())

def transcribe_audio(audio_path):
    try:
        with open(audio_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech_v1.RecognitionAudio(content=content)
        
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,  # Enable word-level timing
            audio_channel_count=1,
            model="video",  # Use video model for better accuracy
            use_enhanced=True  # Use enhanced model
        )

        response = speech_client.recognize(config=config, audio=audio)
        
        transcript = []
        for result in response.results:
            alternative = result.alternatives[0]
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                
                # Format times with millisecond precision
                start_formatted = f"{int(start_time//60):02d}:{start_time%60:06.3f}"
                end_formatted = f"{int(end_time//60):02d}:{end_time%60:06.3f}"
                
                transcript.append({
                    "word": word,
                    "start_time": start_formatted,
                    "end_time": end_formatted
                })
        
        return transcript
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return None

def generate_style_and_edits(transcript, style):
    logger.info(f"Generating style and edits for style: {style}")
    prompt = f"""
    Given the following transcript and style, generate:
    1. A stylized version of the transcript
    2. A list of video editing instructions in JSON format
    
    Transcript: {transcript}
    Style: {style}
    
    The video editing instructions should include cuts, effects, and transitions suitable for the style.
    Each instruction should have a 'type' (e.g., 'cut', 'speed', 'effect'), 'start_time', 'end_time', and 'parameters' (even if empty).
    
    Output the stylized transcript and editing instructions as a JSON object with keys 'stylized_transcript' and 'editing_instructions'.
    Do not include any markdown formatting in your response.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a video editing expert who generates stylized transcripts and editing instructions."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        logger.info(f"Raw API response: {content}")
        
        cleaned_content = clean_json_response(content)
        
        try:
            result = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Error: {str(e)}. Raw content: {cleaned_content}")
            return None
        
        if not isinstance(result, dict) or 'stylized_transcript' not in result or 'editing_instructions' not in result:
            logger.error(f"Invalid response structure. Expected keys missing. Got: {result}")
            return None
        
        # Ensure all edit instructions have a 'parameters' key
        for instruction in result['editing_instructions']:
            if 'parameters' not in instruction:
                instruction['parameters'] = {}
        
        logger.info(f"Generated style and edits: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating style and edits: {str(e)}")
        return None

def apply_edits(video_path, edits):
    try:
        logger.info(f"Applying edits to {video_path}")
        video = VideoFileClip(video_path)
        final_clip = video.copy()
        
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration} seconds")
        
        if not edits:
            logger.warning("No edits provided, returning original video")
            return video_path, []
            
        for edit in edits:
            try:
                edit_type = edit.get('type', 'speed')  # Default to speed if no type
                # Ensure timestamps don't exceed video duration
                start_time = min(time_to_seconds(edit.get('start_time', 0)), video_duration)
                end_time = min(time_to_seconds(edit.get('end_time', video_duration)), video_duration)
                
                # Skip if start_time >= end_time or invalid times
                if start_time >= end_time or start_time >= video_duration:
                    logger.warning(f"Skipping edit: invalid time range {start_time} to {end_time}")
                    continue
                    
                params = edit.get('parameters', {})
                
                logger.info(f"Applying {edit_type} edit from {start_time} to {end_time}")
                
                if edit_type == 'speed':
                    speed_factor = float(str(params.get('speed_factor', '1.5')).replace('x', ''))
                    before_segment = final_clip.subclip(0, start_time) if start_time > 0 else None
                    speed_segment = final_clip.subclip(start_time, end_time).speedx(speed_factor)
                    after_segment = final_clip.subclip(end_time) if end_time < video_duration else None
                    
                    # Combine segments that exist
                    segments = [seg for seg in [before_segment, speed_segment, after_segment] if seg is not None]
                    final_clip = concatenate_videoclips(segments)
                    
            except Exception as edit_error:
                logger.error(f"Error applying edit: {str(edit_error)}")
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        output_path = os.path.join(
            'output', 
            f'edited_{os.path.basename(video_path)}'
        )
        
        logger.info(f"Writing final video to {output_path}")
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            audio=True,
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        video.close()
        final_clip.close()
        
        return output_path, []  # Return empty list for transcripts if not needed
        
    except Exception as e:
        logger.error(f"Error applying edits: {str(e)}")
        if 'video' in locals():
            video.close()
        if 'final_clip' in locals():
            final_clip.close()
        return None, []

def create_unified_timeline(file_transcripts):
    """Create a unified timeline with microsecond precision"""
    current_offset = 0.0
    unified_transcript = []
    
    for transcript in file_transcripts:
        # Add current words with adjusted timestamps
        for word_info in transcript:
            start_time = time_to_seconds(word_info['start_time']) + current_offset
            end_time = time_to_seconds(word_info['end_time']) + current_offset
            
            # Format with microsecond precision
            unified_transcript.append({
                'word': word_info['word'],
                'start_time': f"{int(start_time//60):02d}:{start_time%60:09.6f}",
                'end_time': f"{int(end_time//60):02d}:{end_time%60:09.6f}",
                'original_clip_time': word_info['start_time']
            })
        
        # Update offset with microsecond precision
        last_word = transcript[-1] if transcript else None
        if last_word:
            current_offset += time_to_seconds(last_word['end_time'])
    
    return unified_transcript

def format_transcript_for_openai(transcript_list: List[Dict]) -> str:
    """Convert transcript list to a readable string format with timing info"""
    try:
        formatted_segments = []
        for word_info in transcript_list:
            # Format: "word (MM:SS.mmm - MM:SS.mmm)"
            formatted_segments.append(
                f"{word_info['word']} ({word_info['start_time']} - {word_info['end_time']})"
            )
        
        # Create a more structured format for OpenAI
        formatted_text = (
            "Transcript with timing:\n\n" + 
            " ".join(formatted_segments) +
            "\n\nPlease analyze this content and provide editing instructions in the exact JSON format specified."
        )
        
        logger.info(f"Formatted transcript: {formatted_text[:200]}...")  # Log first 200 chars
        return formatted_text
        
    except Exception as e:
        logger.error(f"Error formatting transcript: {str(e)}")
        return ""

def process_from_gcs(blob_name: str, style: str) -> Tuple[str, List[dict]]:
    """Process a file from Google Cloud Storage"""
    try:
        bucket_name = 'shaz-video-editor-bucket'  # Always use this bucket
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        
        logger.info(f"Processing file from GCS: {gcs_uri}")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob_name)[1]) as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file_path = temp_file.name
        
        # Process the file
        result = process_multiple_files([temp_file_path], style)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing file from GCS: {str(e)}")
        return None, f"Processing from GCS failed: {str(e)}"

def main(file_paths: List[str], style: str, task_id: str = None) -> Tuple[str, List[dict]]:
    """Process multiple video files with unified timeline"""
    try:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        logger.info(f"Processing {len(file_paths)} videos with style: {style}")
        
        # Process all files together with unified timeline
        final_clips, unified_transcript = process_multiple_files(file_paths, style, task_id)
        
        if not final_clips:
            logger.error("Failed to process videos with unified timeline")
            return None, []
            
        # Combine all processed clips
        output_path = combine_videos(final_clips)
        if not output_path:
            logger.error("Failed to combine videos")
            return None, []
            
        return output_path, [unified_transcript]
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return None, []

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_json_response(response):
    cleaned = re.sub(r'^```json\s*', '', response)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned

def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS.NNN format to seconds"""
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
            
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(time_str)
    except Exception as e:
        logger.error(f"Error converting time: {str(e)}")
        return 0.0

def escape_ffmpeg_text(text: str) -> str:
    """
    Robust FFmpeg text escaping for drawtext filter.
    Handles special characters and common issues.
    """
    # Dictionary of special characters and their escaped versions
    escape_chars = {
        "'": r"\'",      # Single quotes
        ":": r"\:",      # Colons
        ",": r"\,",      # Commas
        "[": r"\[",      # Square brackets
        "]": r"\]",
        "%": r"\%",      # Percent signs
        "{": r"\{",      # Curly braces
        "}": r"\}",
        "#": r"\#",      # Hash
        "&": r"\&",      # Ampersand
        "?": r"\?",      # Question mark
        "=": r"\=",      # Equals
        "|": r"\|",      # Pipe
        "<": r"\<",      # Angle brackets
        ">": r"\>",
        "`": r"\`",      # Backtick
        "$": r"\$",      # Dollar sign
        "\\": r"\\",     # Backslash
        "\n": r"\n",     # Newline
        "\r": r"\r",     # Carriage return
        "\t": r"\t"      # Tab
    }
    
    # First pass: escape backslashes
    text = text.replace("\\", "\\\\")
    
    # Second pass: escape all other special characters
    for char, escaped in escape_chars.items():
        if char != "\\":  # Skip backslash as it's already handled
            text = text.replace(char, escaped)
    
    # Handle Unicode characters
    escaped_text = ""
    for char in text:
        if ord(char) < 128:
            escaped_text += char
        else:
            escaped_text += f"\\u{ord(char):04x}"
    
    return escaped_text

def upload_to_gcs(bucket, blob_name, file_path):
    try:
        blob = bucket.blob(blob_name)
        
        with open(file_path, 'rb') as f:
            blob.upload_from_file(
                f,
                content_type='video/mp4',
                timeout=300
            )
        
        url = blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=30),
            method='GET',
            response_type='video/mp4',
            query_parameters={
                'response-content-disposition': f'attachment; filename="edited_video.mp4"',
                'response-content-type': 'video/mp4'
            }
        )
        
        return url
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise

def process_videos_cloud(gcs_uris: List[str], style: str, instructions: dict = None) -> str:
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket('shaz-video-editor-bucket')
        
        with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
            # Download videos
            input_files = []
            for idx, gcs_uri in enumerate(gcs_uris):
                input_path = os.path.join(temp_dir, f"input_{idx}.mov")
                blob_name = gcs_uri.split('/', 3)[-1] if gcs_uri.startswith('gs://') else gcs_uri.lstrip('/')
                bucket.blob(blob_name).download_to_filename(input_path)
                input_files.append(input_path)
            
            # Create concat file
            concat_file = os.path.join(temp_dir, 'concat.txt')
            with open(concat_file, 'w') as f:
                for file in input_files:
                    f.write(f"file '{os.path.abspath(file)}'\n")
            
            # First concatenate videos with better quality settings
            base_output = os.path.join(temp_dir, 'base.mp4')
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-maxrate', '8M',
                '-bufsize', '16M',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                base_output
            ]
            
            logger.info(f"Running concat command: {' '.join(concat_cmd)}")
            subprocess.run(concat_cmd, check=True)
            
            # Use provided instructions or get from editing_instructions.py
            if instructions is None:
                if hasattr(current_app, 'openai_instructions'):
                    instructions = current_app.openai_instructions
                else:
                    raise ValueError("No editing instructions available")
            
            # Apply effects using FFmpeg
            output_path = apply_ai_style(base_output, instructions['segments'])
            
            if not output_path or not os.path.exists(output_path):
                raise ValueError("Failed to generate output video")
            
            # Upload result
            timestamp = int(time.time())
            output_blob_name = f"output/processed_{timestamp}.mp4"
            bucket.blob(output_blob_name).upload_from_filename(output_path)
            
            # Generate signed URL
            url = bucket.blob(output_blob_name).generate_signed_url(
                version='v4',
                expiration=datetime.timedelta(minutes=30),
                method='GET',
                response_type='video/mp4'
            )
            
            return url
            
    except Exception as e:
        logger.error(f"Error in process_videos_cloud: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_multiple_from_gcs(gcs_urls: List[str], style: str, task_id: str) -> Tuple[str, List[str]]:
    try:
        storage_client = storage.Client()
        transcripts = []
        video_durations = []
        total_duration = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download and process videos
            processed_files = []
            for idx, gcs_url in enumerate(gcs_urls):
                try:
                    update_progress(task_id, 'download', 
                                  20 + (idx * 10), 
                                  f'Processing video {idx + 1} of {len(gcs_urls)}')
                    
                    input_path = os.path.join(temp_dir, f"input_{idx}.mov")
                    
                    # Always use shaz-video-editor-bucket
                    bucket_name = 'shaz-video-editor-bucket'
                    
                    # Get blob name from URL
                    if gcs_url.startswith('gs://'):
                        blob_name = gcs_url.split('/', 3)[-1]
                    else:
                        blob_name = gcs_url.lstrip('/')
                    
                    logger.info(f"Accessing bucket: {bucket_name}, blob: {blob_name}")
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(input_path)
                    processed_files.append(input_path)
                    
                    # Get video duration
                    duration = get_video_duration(input_path)
                    video_durations.append(duration)
                    total_duration += duration
                    
                    # Get transcript
                    audio_path = os.path.join(temp_dir, f"audio_{idx}.wav")
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', input_path,
                        '-ac', '1',
                        '-ar', '44100',
                        audio_path
                    ], check=True)
                    
                    transcript = process_single_video_transcript(audio_path, idx, total_duration - duration)
                    if transcript:
                        transcripts.append({
                            'video_idx': idx,
                            'duration': duration,
                            'transcript': transcript
                        })
                    
                    os.unlink(audio_path)
                    
                except Exception as e:
                    logger.error(f"Error processing video {idx}: {str(e)}")
                    continue
            
            # Generate editing instructions
            update_progress(task_id, 'analyze', 60, 'Analyzing content')
            editor = EditingInstructionsGenerator(current_app.config['OPENAI_API_KEY'])
            
            timing_info = {
                'videos': [
                    {
                        'index': idx,
                        'duration': dur,
                        'start_time': sum(video_durations[:idx])
                    } for idx, dur in enumerate(video_durations)
                ],
                'total_duration': total_duration
            }
            
            # Get instructions and store them
            instructions = editor.analyze_content(transcripts, style, timing_info)
            if not instructions:
                raise ValueError("Failed to generate editing instructions")
            
            # Process videos with instructions
            update_progress(task_id, 'edit', 80, 'Processing videos')
            result = process_videos_cloud(gcs_urls, style, instructions)  # Pass instructions here
            
            if not result:
                raise ValueError("Failed to process videos")
            
            update_progress(task_id, 'upload', 90, 'Uploading final video')
            
            return result, transcripts
            
    except Exception as e:
        logger.error(f"Error processing from GCS: {str(e)}")
        update_progress(task_id, 'error', 100, f'Error: {str(e)}')
        return None, []

def process_segment(input_path: str, start_time: float, end_time: float, 
                   segment: dict, temp_dir: str, idx: int) -> str:
    """Process a single video segment with effects"""
    try:
        if end_time <= start_time:
            logger.warning(f"Invalid segment timing: {start_time} to {end_time}")
            return None
            
        effect_type = segment.get('effect_type')
        params = segment.get('parameters', {})
        output_path = os.path.join(temp_dir, f"segment_{idx}.mov")
        
        # Improved FFmpeg base command
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ss', f"{start_time:.3f}",  # More precise timing
            '-t', f"{end_time - start_time:.3f}",
            '-avoid_negative_ts', '1',
            '-copyts'
        ]
        
        filter_complex = []
        
        # Add effect-specific filters
        if effect_type == 'speed_up':
            speed = float(params.get('speed', 1.0))
            filter_complex.extend([
                f'[0:v]setpts={1/speed}*PTS[v]',
                f'[0:a]atempo={min(2.0, speed)}[a]'  # Limit audio speed
            ])
            
        elif effect_type == 'zoom':
            zoom = float(params.get('zoom', 1.0))
            filter_complex.append(
                f'[0:v]scale=iw*{zoom}:-1,crop=iw/zoom:ih/zoom[v]'
            )
            
        elif effect_type == 'text_overlay':
            text = params.get('text', '')
            position = params.get('position', 'center')
            font_size = params.get('font_size', 48)
            
            y_pos = {
                'top': 'h/10',
                'center': '(h-text_h)/2',
                'bottom': 'h-text_h-(h/10)'
            }.get(position, '(h-text_h)/2')
            
            filter_complex.append(
                f'[0:v]drawtext=text=\'{text}\':'
                f'fontsize={font_size}:'
                f'fontcolor=white:'
                f'x=(w-text_w)/2:y={y_pos}:'
                f'box=1:boxcolor=black@0.5:'
                f'boxborderw=5:font=Arial[v]'
            )
            
        elif effect_type == 'fade':
            duration = min(1.0, (end_time - start_time) / 4)  # Limit fade duration
            filter_complex.append(
                f'[0:v]fade=t=in:st=0:d={duration},'
                f'fade=t=out:st={end_time-start_time-duration}:d={duration}[v]'
            )
        
        # Add filter complex if any
        if filter_complex:
            cmd.extend(['-filter_complex', ';'.join(filter_complex)])
            cmd.extend(['-map', '[v]'])
            if '[a]' in ''.join(filter_complex):
                cmd.extend(['-map', '[a]'])
            else:
                cmd.extend(['-map', '0:a'])
        
        # Output settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ])
        
        # Run FFmpeg with proper error handling
        try:
            result = subprocess.run(cmd, 
                                  check=True,
                                  capture_output=True,
                                  text=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                logger.error("Output file is empty or doesn't exist")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return None

def get_video_duration(file_path: str) -> float:
    """Get duration of video file using FFprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return 0.0

def process_single_video_transcript(audio_path: str, video_idx: int, start_offset: float) -> List[dict]:
    """Process transcription for a single video"""
    try:
        with open(audio_path, 'rb') as audio_file:
            content = audio_file.read()
        
        audio = speech_v1.RecognitionAudio(content=content)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        
        transcript = []
        for result in response.results:
            for word_info in result.alternatives[0].words:
                transcript.append({
                    'word': word_info.word,
                    'start_time': word_info.start_time.total_seconds() + start_offset,
                    'end_time': word_info.end_time.total_seconds() + start_offset,
                    'video_idx': video_idx
                })
        
        return transcript
        
    except Exception as e:
        logger.error(f"Error processing video transcript: {str(e)}")
        return []

def process_multiple_files(files: List[str], style: str, task_id: str = None) -> Tuple[str, List[dict]]:
    """Process multiple video files with unified timeline"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process files and get durations
        processed_files, durations = process_files_and_durations(files, temp_dir)
        
        # Get transcripts
        transcripts = []
        current_offset = 0
        for idx, file_path in enumerate(processed_files):
            audio_path = os.path.join(temp_dir, f"audio_{idx}.wav")
            subprocess.run([
                'ffmpeg', '-y',
                '-i', file_path,
                '-ac', '1',
                '-ar', '44100',
                audio_path
            ], check=True)
            
            transcript = process_single_video_transcript(audio_path, idx, current_offset)
            if transcript:
                transcripts.append({
                    'video_idx': idx,
                    'duration': durations[idx],
                    'transcript': transcript
                })
            current_offset += durations[idx]
            os.unlink(audio_path)
        
        # Generate editing instructions with API key from environment
        editor = EditingInstructionsGenerator(api_key=os.getenv('OPENAI_API_KEY'))
        timing_info = {
            'videos': [
                {
                    'index': idx,
                    'duration': dur,
                    'start_time': sum(durations[:idx])
                } for idx, dur in enumerate(durations)
            ],
            'total_duration': sum(durations)
        }
        
        instructions = editor.analyze_content(transcripts, style, timing_info)
        if not instructions:
            raise ValueError("Failed to generate editing instructions")
            
        # Process with instructions
        result = process_videos_cloud(files, style)
        return result, transcripts

def extract_audio_and_duration(video_path: str, audio_path: str) -> float:
    """Extract audio and get duration in one FFmpeg pass"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # No video
            '-ac', '1',  # Mono audio
            '-ar', '44100',  # Sample rate
            audio_path
        ]
        
        # Run FFmpeg and capture duration from stderr
        result = subprocess.run(cmd, 
                              capture_output=True,
                              text=True,
                              check=True)
        
        # Get duration using ffprobe
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        
        return duration
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error in extract_audio_and_duration: {str(e)}")
        raise

def process_files_and_durations(files: List[str], temp_dir: str) -> Tuple[List[str], List[float]]:
    """Process files and return their paths and durations"""
    processed_files = []
    durations = []
    
    for idx, file_path in enumerate(files):
        try:
            output_path = os.path.join(temp_dir, f"processed_{idx}.mov")
            # Normalize video format
            subprocess.run([
                'ffmpeg', '-y',
                '-i', file_path,
                '-c:v', 'h264',
                '-c:a', 'aac',
                '-pix_fmt', 'yuv420p',
                output_path
            ], check=True)
            
            duration = get_video_duration(output_path)
            processed_files.append(output_path)
            durations.append(duration)
            
        except Exception as e:
            logger.error(f"Error processing file {idx}: {str(e)}")
            continue
            
    return processed_files, durations

def verify_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True)
        logger.info(f"FFmpeg version: {result.stdout.split()[2]}")
        return True
    except Exception as e:
        logger.error(f"FFmpeg not found: {str(e)}")
        return False

# Add check at startup
if not verify_ffmpeg():
    raise RuntimeError("FFmpeg not available in container")

if __name__ == "__main__":
    # Your main execution code here if needed
    pass

