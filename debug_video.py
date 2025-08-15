#!/usr/bin/env python3
"""
Debug script to test video processing capabilities
"""
import subprocess
import json
import sys
import os

def test_video_info(video_path):
    """Test getting video information"""
    print(f"Testing video info for: {video_path}")
    print("-" * 50)
    
    if not os.path.exists(video_path):
        print(f"ERROR: File does not exist: {video_path}")
        return False
    
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        print("FORMAT INFO:")
        format_info = data.get('format', {})
        print(f"  Duration: {format_info.get('duration', 'Unknown')}")
        print(f"  Size: {format_info.get('size', 'Unknown')} bytes")
        print(f"  Format: {format_info.get('format_name', 'Unknown')}")
        
        print("\nSTREAMS:")
        for i, stream in enumerate(data.get('streams', [])):
            print(f"  Stream {i}:")
            print(f"    Type: {stream.get('codec_type', 'Unknown')}")
            print(f"    Codec: {stream.get('codec_name', 'Unknown')}")
            if stream.get('codec_type') == 'video':
                print(f"    Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                print(f"    Frame rate: {stream.get('r_frame_rate', 'Unknown')}")
                print(f"    Duration: {stream.get('duration', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR getting video info: {e}")
        return False

def test_simple_extract(video_path, start_time=10, duration=5):
    """Test simple video extraction"""
    print(f"\nTesting simple extraction: {start_time}s for {duration}s")
    print("-" * 50)
    
    output_path = "test_extract.mp4"
    
    # Get video info first to determine codec
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_codec = 'unknown'
        has_audio = False
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_codec = stream.get('codec_name', 'unknown')
            elif stream.get('codec_type') == 'audio':
                has_audio = True
        
        print(f"Detected: Video codec={video_codec}, Audio={'Yes' if has_audio else 'No'}")
        
    except:
        video_codec = 'unknown'
        has_audio = True
        print("Could not detect codec info, assuming defaults")
    
    # Choose encoding strategy based on codec
    if video_codec in ['vp9', 'vp8']:
        video_params = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        print("Using VP9/VP8 compatible encoding...")
    else:
        video_params = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
        print("Using standard H.264 encoding...")
    
    # Try with audio if available
    if has_audio:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time), '-t', str(duration)
        ] + video_params + [
            '-c:a', 'aac', '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            '-y', output_path
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Extraction with audio worked")
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"Output file size: {size} bytes")
                os.unlink(output_path)  # Clean up
            return True
        else:
            print("FAILED: Extraction with audio failed")
            print(f"Error: {result.stderr}")
    
    # Try without audio
    print("Trying without audio...")
    cmd = [
        'ffmpeg', '-i', video_path,
        '-ss', str(start_time), '-t', str(duration)
    ] + video_params + [
        '-an',
        '-avoid_negative_ts', 'make_zero',
        '-y', output_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS: Extraction without audio worked")
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"Output file size: {size} bytes")
            os.unlink(output_path)  # Clean up
        return True
    else:
        print("FAILED: Extraction without audio also failed")
        print(f"Error: {result.stderr}")
        
        # Try stream copy as last resort
        print("Trying stream copy method...")
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time), '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y', output_path
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Stream copy method worked")
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"Output file size: {size} bytes")
                os.unlink(output_path)  # Clean up
            return True
        else:
            print("FAILED: Stream copy also failed")
            print(f"Error: {result.stderr}")
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_video.py <video_file_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print("VIDEO PROCESSING DEBUG TOOL")
    print("=" * 50)
    
    # Test 1: Get video info
    success1 = test_video_info(video_path)
    
    # Test 2: Simple extraction
    success2 = test_simple_extract(video_path)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Video info test: {'PASS' if success1 else 'FAIL'}")
    print(f"Extraction test: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("✅ Video should work with the processor")
    else:
        print("❌ Video may have issues with the processor")

if __name__ == "__main__":
    main()
