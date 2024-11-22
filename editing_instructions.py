# editing_instructions.py

# Add MoviePy imports at the top
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, AudioFileClip, TextClip, CompositeVideoClip, transfx

# Define constants for edit types
TEXT_OVERLAY = 'text_overlay'
TRANSITION = 'transition'

# Add text styles configuration
TEXT_STYLES = {
    'title': {
        'fontsize': 70,
        'color': 'white',
        'font': 'Arial-Bold',
        'stroke_color': 'black',
        'stroke_width': 2,
        'method': 'label'
    },
    'subtitle': {
        'fontsize': 40,
        'color': 'white',
        'font': 'Arial',
        'stroke_color': 'black',
        'stroke_width': 1,
        'method': 'label'
    }
}

# Add transition types
TRANSITION_TYPES = {
    'fade': lambda clip, duration: clip.crossfadein(duration),
    'slide_in': lambda clip, duration: clip.slide_in(duration=duration, side='right'),
    'slide_out': lambda clip, duration: clip.slide_out(duration=duration, side='left'),
    'crossfade': lambda clip, duration: clip.crossfadein(duration)
}

# First, set up logging
import logging
logger = logging.getLogger(__name__)

# Then other imports
import openai
from typing import List, Dict, Any
import json
from editing_presets import VideoStyle, STYLE_DEFINITIONS
from openai import OpenAI
import re
import requests
from moviepy.audio.AudioClip import CompositeAudioClip
import tempfile
import os
import traceback
import math
import numpy as np
import time
from google.cloud import videointelligence
import subprocess
import shutil
import uuid

# Add this class at the top of the file after imports
class EditingInstructionsGenerator:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)

    def analyze_content(self, transcripts: List[dict], style: str, timing_info: dict) -> Dict[str, Any]:
        try:
            # Get master prompt
            prompt = create_master_prompt(timing_info, style, transcripts)
            
            # Use only one OpenAI call with lower temperature
            messages = [
                {"role": "system", "content": "You are a precise video editing assistant. Generate exact timings and effects."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,  # Slightly higher for creativity
                max_tokens=1000,
                presence_penalty=-0.5,
                frequency_penalty=0.0
            )
            
            content = response.choices[0].message.content.strip()
            self.logger.info(f"OpenAI Raw Response: {content}")
            
            try:
                parsed_content = json.loads(content)
                if "segments" not in parsed_content:
                    raise ValueError("Response missing 'segments' key")
                    
                return parsed_content
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}")
                self.logger.error(f"Content that failed to parse: {content}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return None

def _time_to_seconds(time_str: str) -> float:
    """Convert various time formats to seconds"""
    try:
        # If already a number, return it
        if isinstance(time_str, (int, float)):
            return float(time_str)
            
        # Handle MM:SS.mmm format
        if ':' in str(time_str):
            minutes, seconds = str(time_str).split(':')
            total_seconds = float(minutes) * 60
            if '.' in seconds:
                seconds, milliseconds = seconds.split('.')
                total_seconds += float(seconds) + float(f"0.{milliseconds}")
            else:
                total_seconds += float(seconds)
            return total_seconds
            
        # Handle SS.mmm format
        if '.' in str(time_str):
            seconds, milliseconds = str(time_str).split('.')
            return float(seconds) + float(f"0.{milliseconds}")
            
        # Handle plain seconds
        return float(time_str)
        
    except Exception as e:
        logger.error(f"Error converting time {time_str}: {str(e)}")
        return 0.0

def _apply_effect(clip: VideoFileClip, effect: Dict[str, Any]) -> VideoFileClip:
    """Apply a single effect to a clip"""
    if not clip or clip.duration == 0:
        logger.error("Invalid clip or zero duration")
        return clip
        
    effect_name = effect.get('name', effect.get('effect_type', ''))
    params = effect.get('parameters', {})
    intensity = params.get('intensity', 'medium')
    
    logger.info(f"Applying effect: {effect_name}")
    logger.info(f"Effect parameters: {params}")
    
    try:
        if effect_name == 'text_overlay':
            text = params.get('text', '')
            style = params.get('style', 'subtitle')
            position = params.get('position', 'bottom')
            
            # Create text clip with style
            text_style = TEXT_STYLES[style].copy()
            txt_clip = TextClip(
                text,
                fontsize=text_style['fontsize'],
                color=text_style['color'],
                font=text_style['font'],
                stroke_color=text_style['stroke_color'],
                stroke_width=text_style['stroke_width'],
                method=text_style['method']
            )
            
            # Position the text
            if position == 'bottom':
                txt_pos = ('center', 0.8)  # 80% down the screen
            elif position == 'top':
                txt_pos = ('center', 0.1)  # 10% from the top
            else:
                txt_pos = 'center'
                
            # Add text duration and fade
            txt_clip = (txt_clip
                       .set_duration(clip.duration)
                       .set_position(txt_pos)
                       .fadein(0.5)
                       .fadeout(0.5))
            
            # Combine with video
            return CompositeVideoClip([clip, txt_clip])
            
        elif effect_name == 'transition':
            transition_type = params.get('type', 'fade')
            duration = min(params.get('duration', 1.0), clip.duration / 2)
            
            if transition_type == 'fade':
                return clip.fadein(duration).fadeout(duration)
            elif transition_type == 'slide_in':
                return clip.fx(transfx.slide_in, duration=duration, side='right')
            elif transition_type == 'slide_out':
                return clip.fx(transfx.slide_out, duration=duration, side='left')
            elif transition_type == 'crossfade':
                return clip.crossfadein(duration).crossfadeout(duration)
            else:
                logger.warning(f"Unknown transition type: {transition_type}")
                return clip
                
        elif effect_name == 'speed_up':
            speed_factor = {
                'low': 1.25,
                'medium': 1.5,
                'high': 2.0
            }.get(intensity, 1.5)
            
            # Apply speed effect with audio preservation
            return clip.fx(vfx.speedx, speed_factor)
            
        elif effect_name == 'zoom':
            zoom_factor = {
                'low': 1.1,
                'medium': 1.2,
                'high': 1.3
            }.get(intensity, 1.2)
            
            # Create smooth zoom effect
            def zoom(t):
                # Calculate zoom progress (0 to 1)
                progress = t / clip.duration
                # Apply smooth easing
                eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
                # Calculate current zoom level
                current_zoom = 1 + (zoom_factor - 1) * eased_progress
                return current_zoom
                
            return clip.fx(vfx.resize, zoom)
            
        elif effect_name == 'fade':
            duration = min(params.get('duration', 0.5), clip.duration / 2)
            return clip.fadein(duration).fadeout(duration)
            
    except Exception as e:
        logger.error(f"Error applying effect {effect_name}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return clip
    
    return clip

def apply_ai_style(input_path: str, edits: list) -> str:
    try:
        temp_dir = tempfile.mkdtemp(dir='/tmp')
        output_path = os.path.join(temp_dir, 'output.mp4')
        
        # Build filter chains
        video_parts = []
        audio_parts = []
        
        # First pass: Create segments
        for i, edit in enumerate(edits):
            start_time = _time_to_seconds(edit['start_time'])
            end_time = _time_to_seconds(edit['end_time'])
            
            if edit['effect_type'] == 'speed_up':
                speed = float(edit['parameters'].get('speed', 2.0))
                video_parts.append(
                    f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS*{1/speed}[v{i}];"
                )
                audio_parts.append(
                    f"[0:a]atrim=start={start_time}:end={end_time},atempo={speed}[a{i}];"
                )
                
            elif edit['effect_type'] == 'zoom':
                video_parts.append(
                    f"[0:v]trim=start={start_time}:end={end_time},"
                    f"scale=iw*{edit['parameters'].get('zoom', 1.5)}:-1,"
                    f"crop=iw/{edit['parameters'].get('zoom', 1.5)}:ih/{edit['parameters'].get('zoom', 1.5)}[v{i}];"
                )
                # Copy corresponding audio
                audio_parts.append(
                    f"[0:a]atrim=start={start_time}:end={end_time}[a{i}];"
                )
                
            elif edit['effect_type'] == 'fade':
                duration = 0.5
                video_parts.append(
                    f"[0:v]trim=start={start_time}:end={end_time},"
                    f"fade=t=in:st=0:d={duration},"
                    f"fade=t=out:st={end_time-start_time-duration}:d={duration}[v{i}];"
                )
                audio_parts.append(
                    f"[0:a]atrim=start={start_time}:end={end_time},"
                    f"afade=t=in:st=0:d={duration},"
                    f"afade=t=out:st={end_time-start_time-duration}:d={duration}[a{i}];"
                )
        
        # Build final filter complex
        filter_complex = ''.join(video_parts)
        filter_complex += ''.join(audio_parts)
        
        # Add concat if we have segments
        if video_parts:
            video_inputs = ''.join(f'[v{i}]' for i in range(len(video_parts)))
            audio_inputs = ''.join(f'[a{i}]' for i in range(len(audio_parts)))
            
            filter_complex += (
                f"{video_inputs}concat=n={len(video_parts)}:v=1[vout];"
                f"{audio_inputs}concat=n={len(audio_parts)}:a=1[aout]"
            )
        else:
            filter_complex = "[0:v]null[vout];[0:a]anull[aout]"
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-filter_complex', filter_complex,
            '-map', '[vout]',
            '-map', '[aout]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return output_path
        
    except Exception as e:
        logger.error(f"Error applying effects: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def validate_parameters(effect_type: str, params: dict) -> dict:
    """Validate and clamp parameters to acceptable ranges"""
    validated = params.copy()
    
    if effect_type == 'speed_up':
        speed = float(params.get('speed', 2.0))
        validated['speed'] = max(0.5, min(4.0, speed))
        
    elif effect_type == 'zoom':
        zoom = float(params.get('zoom', 1.5))
        validated['zoom'] = max(1.1, min(3.0, zoom))
        
    elif effect_type == 'text_overlay':
        font_size = int(params.get('font_size', 48))
        validated['font_size'] = max(24, min(72, font_size))
        validated['position'] = params.get('position', 'center')
        if validated['position'] not in ['top', 'center', 'bottom']:
            validated['position'] = 'center'
            
        # Ensure text is a string
        validated['text'] = str(params.get('text', ''))
            
    return validated

# Add at the top of the file
def create_master_prompt(timing_info: dict, style: str, transcripts: List[dict]) -> str:
    """Create a master prompt that includes FFmpeg filter syntax examples"""
    
    # Format transcript content
    transcript_content = []
    for t in transcripts:
        if 'transcript' in t:
            words = [w.get('word', '') for w in t.get('transcript', [])]
            transcript_content.append(' '.join(words))
    
    # Build prompt parts
    prompt_parts = [
        f"Video Editor Task",
        f"Total Duration: {timing_info['total_duration']:.1f} seconds",
        f"Style: {style}",
        "",
        "Content Analysis:",
        "Available transcripts:",
        *transcript_content,
        "",
        "Task: Generate FFmpeg filter commands for video editing.",
        "",
        "Available Effects and Their FFmpeg Syntax:",
        "",
        "1. Text Overlay:",
        "   FFmpeg syntax: drawtext=text='ACTUAL_TEXT':fontsize=SIZE:fontcolor=white:x=(w-text_w)/2:y=POS:box=1:boxcolor=black@0.5",
        "   - SIZE: 24-72",
        "   - POS: h/10 (top), (h-text_h)/2 (center), 9*h/10-text_h (bottom)",
        "   - Duration: 2-4 seconds",
        "",
        "2. Speed Effect:",
        "   FFmpeg syntax: setpts=PTS*SPEED_FACTOR;atempo=AUDIO_SPEED",
        "   - SPEED_FACTOR: 0.25-2.0 (0.5 = 2x speed)",
        "   - AUDIO_SPEED: must match video speed",
        "   - Duration: 2-5 seconds",
        "",
        "3. Zoom Effect:",
        "   FFmpeg syntax: scale=iw*ZOOM:-1,crop=iw/ZOOM:ih/ZOOM",
        "   - ZOOM: 1.1-3.0",
        "   - Duration: 1-3 seconds",
        "",
        "4. Fade Effect:",
        "   FFmpeg syntax: fade=t=TYPE:st=START:d=DURATION",
        "   - TYPE: in/out",
        "   - DURATION: exactly 0.5 seconds",
        "",
        "CRITICAL RULES:",
        "1. Maximum 8-10 total segments",
        "2. Effects must not overlap",
        "3. Use exact FFmpeg syntax",
        "4. Text must match current audio",
        "5. Effects should enhance content",
        "",
        "Response Format:",
        "{",
        "  \"segments\": [",
        "    {",
        "      \"start_time\": \"MM:SS.NNN\",",
        "      \"end_time\": \"MM:SS.NNN\",",
        "      \"effect_type\": \"text_overlay|speed_up|zoom|fade\",",
        "      \"ffmpeg_filter\": \"EXACT_FFMPEG_COMMAND\",",
        "      \"parameters\": {",
        "        \"speed\": float,",
        "        \"zoom\": float,",
        "        \"text\": \"actual transcript text\",",
        "        \"position\": \"top|center|bottom\",",
        "        \"font_size\": integer",
        "      }",
        "    }",
        "  ]",
        "}"
    ]
    
    return "\n".join(prompt_parts)

