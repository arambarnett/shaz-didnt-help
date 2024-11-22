# AI Video Editor

## Overview
A Flask-based web application that uses AI to automatically edit videos. The application:
1. Takes multiple video uploads
2. Uses OpenAI to analyze content
3. Applies automated effects (speed changes, zooms, fades)
4. Processes using FFmpeg
5. Stores videos in Google Cloud Storage

## Current Issues
1. Video Output Issues:
   - No audio in final output
   - Audio sync problems
   - Effect timing incorrect
   - Duration calculation errors

2. Processing Problems:
   - FFmpeg filter chain errors
   - Stream mapping inconsistencies
   - Effect application failures
   - Style application not working

## Setup Requirements

### Prerequisites
- Python 3.10+
- FFmpeg 7.1
- Google Cloud Storage account
- OpenAI API key
- Docker

### Environment Setup
1. Create `.env` file: 