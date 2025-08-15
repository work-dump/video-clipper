import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import subprocess
import json
import tempfile
import threading
from pathlib import Path
import sys
import cv2
import numpy as np
from datetime import datetime, timedelta
import librosa
import soundfile as sf

class VideoCallProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Call Recording Processor")
        self.root.geometry("800x700")
        
        # Configuration variables
        self.selected_folder = tk.StringVar()
        self.audio_threshold = tk.DoubleVar(value=0.01)  # Audio activity threshold
        self.visual_threshold = tk.DoubleVar(value=0.001)  # Visual change threshold - very sensitive for typing detection
        self.min_inactive_duration = tk.DoubleVar(value=3.0)  # Minimum inactive duration to remove (seconds)
        self.processing = False
        
        # Supported video formats
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp', '.ogv']
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Folder selection
        ttk.Label(main_frame, text="Select Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.selected_folder, state="readonly")
        self.folder_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(folder_frame, text="Browse", command=self.select_folder).grid(row=0, column=1)
        
        # Configuration parameters
        config_frame = ttk.LabelFrame(main_frame, text="Activity Detection Settings", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        config_frame.columnconfigure(1, weight=1)
        
        # Audio threshold
        ttk.Label(config_frame, text="Audio Activity Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        audio_entry = ttk.Entry(config_frame, textvariable=self.audio_threshold, width=10)
        audio_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        ttk.Label(config_frame, text="(0.001-0.1, lower = more sensitive)").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Visual threshold
        ttk.Label(config_frame, text="Visual Change Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        visual_entry = ttk.Entry(config_frame, textvariable=self.visual_threshold, width=10)
        visual_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        ttk.Label(config_frame, text="(0.001-0.01, detects single pixel changes like typing)").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Minimum inactive duration
        ttk.Label(config_frame, text="Min Inactive Duration (sec):").grid(row=2, column=0, sticky=tk.W, pady=2)
        duration_entry = ttk.Entry(config_frame, textvariable=self.min_inactive_duration, width=10)
        duration_entry.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        ttk.Label(config_frame, text="(minimum seconds of inactivity to remove)").grid(row=2, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        
        # Progress section
        ttk.Label(main_frame, text="Progress:").grid(row=2, column=0, sticky=tk.W, pady=(20, 5))
        
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 5))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Video Files", 
                                       command=self.start_processing, state="disabled")
        self.process_button.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Log area
        ttk.Label(main_frame, text="Processing Log:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state="disabled")
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder.set(folder)
            self.log(f"Selected folder: {folder}")
            # Check for video files
            video_files = self.get_video_files(folder)
            if video_files:
                self.log(f"Found {len(video_files)} video files")
                formats_found = set(os.path.splitext(f)[1].lower() for f in video_files)
                self.log(f"Formats found: {', '.join(sorted(formats_found))}")
                self.process_button.config(state="normal")
            else:
                self.log("No supported video files found in selected folder")
                self.process_button.config(state="disabled")
    
    def get_video_files(self, folder):
        video_files = []
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in self.supported_formats):
                video_files.append(os.path.join(folder, file))
        return video_files
    
    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()
    
    def start_processing(self):
        if self.processing:
            return
            
        if not self.selected_folder.get():
            messagebox.showerror("Error", "Please select a folder first")
            return
            
        # Check if FFmpeg is available
        if not self.check_ffmpeg():
            messagebox.showerror("Error", 
                               "FFmpeg is required but not found.\n"
                               "Please install FFmpeg from https://ffmpeg.org/download.html")
            return
        
        self.processing = True
        self.process_button.config(state="disabled")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()
    
    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def process_files(self):
        try:
            folder = self.selected_folder.get()
            video_files = self.get_video_files(folder)
            
            if not video_files:
                self.log("No video files found")
                return
            
            # Create output folder
            folder_name = os.path.basename(folder.rstrip(os.sep))
            output_folder = os.path.join(os.path.dirname(folder), f"{folder_name}_processed")
            os.makedirs(output_folder, exist_ok=True)
            self.log(f"Created output folder: {output_folder}")
            
            # Create logs folder
            logs_folder = os.path.join(output_folder, "logs")
            os.makedirs(logs_folder, exist_ok=True)
            
            total_files = len(video_files)
            self.progress_bar.config(maximum=total_files)
            
            for i, video_file in enumerate(video_files):
                self.progress_var.set(f"Processing {i+1}/{total_files}: {os.path.basename(video_file)}")
                self.progress_bar.config(value=i)
                
                try:
                    self.process_single_file(video_file, output_folder, logs_folder)
                except Exception as e:
                    self.log(f"Error processing {os.path.basename(video_file)}: {str(e)}")
                
                self.root.update_idletasks()
            
            self.progress_bar.config(value=total_files)
            self.progress_var.set(f"Completed! Processed {total_files} files")
            self.log("Processing completed!")
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.processing = False
            self.process_button.config(state="normal")
    
    def process_single_file(self, input_file, output_folder, logs_folder):
        filename = os.path.basename(input_file)
        filename_no_ext = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{filename_no_ext}_processed.mp4")
        log_file = os.path.join(logs_folder, f"{filename_no_ext}_log.txt")
        
        self.log(f"Processing: {filename}")
        
        # Initialize log content
        log_content = []
        log_content.append(f"Processing Log for: {filename}")
        log_content.append(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append(f"Settings: Audio Threshold={self.audio_threshold.get()}, Visual Threshold={self.visual_threshold.get()}, Min Inactive Duration={self.min_inactive_duration.get()}s")
        log_content.append("-" * 80)
        
        try:
            # Analyze video for activity
            self.log(f"Analyzing activity in {filename}...")
            active_segments = self.detect_activity(input_file, log_content)
            
            if not active_segments:
                self.log(f"No active segments found in {filename}")
                log_content.append("No active segments detected. File not processed.")
                self.save_log(log_file, log_content)
                return
            
            # Create output video with only active segments
            self.log(f"Creating processed video with {len(active_segments)} active segments...")
            success = self.create_processed_video(input_file, output_file, active_segments, log_content)
            
            if success:
                self.log(f"Successfully processed: {filename} -> {filename_no_ext}_processed.mp4")
                log_content.append(f"Successfully created processed video: {filename_no_ext}_processed.mp4")
            else:
                self.log(f"Failed to create processed video for {filename}")
                log_content.append("Failed to create processed video.")
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            self.log(error_msg)
            log_content.append(f"ERROR: {str(e)}")
                
        finally:
            log_content.append(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.save_log(log_file, log_content)
    
    def save_log(self, log_file, log_content):
        """Save processing log to file"""
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
        except Exception as e:
            self.log(f"Failed to save log file: {str(e)}")
    
    def detect_activity(self, input_file, log_content):
        """Detect active segments in video based on audio and visual changes"""
        try:
            # Get video information
            video_info = self.get_video_info(input_file)
            if not video_info:
                return []
            
            duration = video_info['duration']
            fps = video_info['fps']
            
            log_content.append(f"Video duration: {duration:.2f} seconds")
            log_content.append(f"Video FPS: {fps:.2f}")
            log_content.append("")
            
            # Analyze audio activity
            self.log("Analyzing audio activity...")
            audio_active_segments = self.analyze_audio_activity(input_file, duration, log_content)
            
            # Analyze visual activity
            self.log("Analyzing visual changes...")
            visual_active_segments = self.analyze_visual_activity(input_file, duration, fps, log_content)
            
            # Combine audio and visual analysis
            combined_segments = self.combine_activity_segments(
                audio_active_segments, visual_active_segments, duration, log_content
            )
            
            return combined_segments
            
        except Exception as e:
            log_content.append(f"Error in activity detection: {str(e)}")
            raise
    
    def get_video_info(self, input_file):
        """Get video information using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            audio_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                self.log("No video stream found in file")
                return None
            
            # Get duration from format or video stream
            duration = None
            if 'duration' in data['format']:
                duration = float(data['format']['duration'])
            elif 'duration' in video_stream:
                duration = float(video_stream['duration'])
            else:
                self.log("Could not determine video duration")
                return None
            
            # Handle frame rate safely
            fps = 30.0  # Default fallback
            if 'r_frame_rate' in video_stream:
                try:
                    fps_str = video_stream['r_frame_rate']
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den) if float(den) != 0 else 30.0
                    else:
                        fps = float(fps_str)
                except (ValueError, ZeroDivisionError):
                    fps = 30.0
            elif 'avg_frame_rate' in video_stream:
                try:
                    fps_str = video_stream['avg_frame_rate']
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den) if float(den) != 0 else 30.0
                    else:
                        fps = float(fps_str)
                except (ValueError, ZeroDivisionError):
                    fps = 30.0
            
            info = {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream.get('width', 1920)),
                'height': int(video_stream.get('height', 1080)),
                'has_audio': audio_stream is not None,
                'video_codec': video_stream.get('codec_name', 'unknown'),
                'audio_codec': audio_stream.get('codec_name', 'none') if audio_stream else 'none'
            }
            
            self.log(f"Video info: {duration:.2f}s, {fps:.2f}fps, {info['width']}x{info['height']}, "
                    f"video:{info['video_codec']}, audio:{info['audio_codec']}")
            
            return info
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.log(f"Failed to get video info: {str(e)}")
            return None
    
    def analyze_audio_activity(self, input_file, duration, log_content):
        """Analyze audio for speech/sound activity"""
        try:
            # Extract audio using FFmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            cmd = [
                'ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', '-y', temp_audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                log_content.append("Warning: Could not extract audio for analysis")
                log_content.append(f"FFmpeg audio extraction error: {result.stderr}")
                # Check if it's because there's no audio track
                if "does not contain any stream" in result.stderr or "No audio" in result.stderr:
                    log_content.append("Video file has no audio track - skipping audio analysis")
                return []
            
            # Load audio with librosa
            y, sr = librosa.load(temp_audio_path, sr=16000)
            
            # Calculate RMS energy in windows
            frame_length = int(sr * 0.5)  # 0.5 second windows
            hop_length = int(sr * 0.1)    # 0.1 second hop
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            
            # Find active segments
            threshold = self.audio_threshold.get()
            active_segments = []
            current_start = None
            
            for i, (time, energy) in enumerate(zip(times, rms)):
                if energy > threshold:
                    if current_start is None:
                        current_start = time
                else:
                    if current_start is not None:
                        active_segments.append((current_start, time))
                        current_start = None
            
            # Close final segment if needed
            if current_start is not None:
                active_segments.append((current_start, duration))
            
            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            log_content.append(f"Audio analysis: Found {len(active_segments)} active audio segments")
            total_active_audio = sum(end - start for start, end in active_segments)
            log_content.append(f"Total active audio time: {total_active_audio:.2f} seconds ({total_active_audio/duration*100:.1f}%)")
            
            return active_segments
            
        except Exception as e:
            log_content.append(f"Audio analysis failed: {str(e)}")
            return []
    
    def analyze_visual_activity(self, input_file, duration, fps, log_content):
        """Analyze video for visual changes - highly sensitive to detect even single pixel changes like typing"""
        try:
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                log_content.append("Warning: Could not open video for visual analysis")
                return []
            
            # Use more frequent sampling for better detection (every 0.2 seconds)
            sample_interval = max(1, int(fps * 0.2))
            threshold = self.visual_threshold.get()
            
            prev_frame = None
            active_segments = []
            current_start = None
            frame_count = 0
            
            # Statistics for logging
            total_changes_detected = 0
            max_change_detected = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    # Convert to grayscale but keep higher resolution for better sensitivity
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Use higher resolution (640x480) to catch subtle changes like cursor movements
                    gray = cv2.resize(gray, (640, 480))
                    
                    if prev_frame is not None:
                        # Multiple change detection methods for maximum sensitivity
                        
                        # Method 1: Absolute difference (catches any pixel change)
                        diff = cv2.absdiff(prev_frame, gray)
                        
                        # Method 2: Count pixels that changed by any amount
                        changed_pixels = np.count_nonzero(diff)
                        total_pixels = gray.shape[0] * gray.shape[1]
                        pixel_change_ratio = changed_pixels / total_pixels
                        
                        # Method 3: Average intensity change
                        mean_change = np.mean(diff) / 255.0
                        
                        # Method 4: Maximum change (catches bright changes like cursor)
                        max_change = np.max(diff) / 255.0
                        
                        # Combine all methods - if ANY method detects change above threshold, consider it active
                        is_active = (
                            pixel_change_ratio > threshold or  # Any pixels changed
                            mean_change > threshold or         # Average change
                            max_change > (threshold * 10)      # Strong local change (like cursor/typing)
                        )
                        
                        current_time = frame_count / fps
                        
                        if is_active:
                            total_changes_detected += 1
                            max_change_detected = max(max_change_detected, max_change)
                            
                            if current_start is None:
                                current_start = current_time
                        else:
                            if current_start is not None:
                                active_segments.append((current_start, current_time))
                                current_start = None
                    
                    prev_frame = gray
                
                frame_count += 1
            
            # Close final segment if needed
            if current_start is not None:
                active_segments.append((current_start, duration))
            
            cap.release()
            
            # Enhanced logging with sensitivity statistics
            log_content.append(f"Visual analysis: Found {len(active_segments)} active visual segments")
            log_content.append(f"Total frames analyzed: {frame_count // sample_interval}")
            log_content.append(f"Changes detected: {total_changes_detected}")
            log_content.append(f"Maximum change detected: {max_change_detected:.4f}")
            
            total_active_visual = sum(end - start for start, end in active_segments)
            log_content.append(f"Total active visual time: {total_active_visual:.2f} seconds ({total_active_visual/duration*100:.1f}%)")
            
            return active_segments
            
        except Exception as e:
            log_content.append(f"Visual analysis failed: {str(e)}")
            return []
    
    def combine_activity_segments(self, audio_segments, visual_segments, duration, log_content):
        """Combine audio and visual activity segments"""
        try:
            # Merge all segments
            all_segments = audio_segments + visual_segments
            if not all_segments:
                return []
            
            # Sort by start time
            all_segments.sort(key=lambda x: x[0])
            
            # Merge overlapping segments
            merged = []
            for start, end in all_segments:
                if merged and start <= merged[-1][1]:
                    # Overlapping, extend the previous segment
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            
            # Filter out segments shorter than minimum duration
            min_duration = self.min_inactive_duration.get()
            final_segments = []
            
            for i, (start, end) in enumerate(merged):
                segment_duration = end - start
                if segment_duration >= min_duration:
                    final_segments.append((start, end))
            
            # Log removed segments
            log_content.append("")
            log_content.append("REMOVED INACTIVE SEGMENTS:")
            log_content.append("-" * 40)
            
            if not final_segments:
                log_content.append(f"All segments removed (entire video is inactive)")
                return []
            
            # Find gaps (inactive periods)
            removed_segments = []
            current_pos = 0
            
            for start, end in final_segments:
                if start > current_pos:
                    # There's a gap before this segment
                    gap_duration = start - current_pos
                    if gap_duration >= min_duration:
                        removed_segments.append((current_pos, start, gap_duration))
                        log_content.append(f"  {self.format_time(current_pos)} - {self.format_time(start)} ({gap_duration:.2f}s): No activity detected")
                current_pos = end
            
            # Check for gap at the end
            if current_pos < duration:
                gap_duration = duration - current_pos
                if gap_duration >= min_duration:
                    removed_segments.append((current_pos, duration, gap_duration))
                    log_content.append(f"  {self.format_time(current_pos)} - {self.format_time(duration)} ({gap_duration:.2f}s): No activity detected")
            
            total_removed = sum(duration for _, _, duration in removed_segments)
            total_kept = sum(end - start for start, end in final_segments)
            
            log_content.append("")
            log_content.append("SUMMARY:")
            log_content.append(f"  Original duration: {self.format_time(duration)} ({duration:.2f}s)")
            log_content.append(f"  Active segments: {len(final_segments)}")
            log_content.append(f"  Total active time: {self.format_time(total_kept)} ({total_kept:.2f}s)")
            log_content.append(f"  Total removed time: {self.format_time(total_removed)} ({total_removed:.2f}s)")
            log_content.append(f"  Time saved: {total_removed/duration*100:.1f}%")
            
            return final_segments
            
        except Exception as e:
            log_content.append(f"Error combining segments: {str(e)}")
            return []
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def create_processed_video(self, input_file, output_file, active_segments, log_content):
        """Create output video with only active segments"""
        try:
            if not active_segments:
                log_content.append("No active segments to process")
                return False
            
            log_content.append(f"Creating video with {len(active_segments)} segments:")
            for i, (start, end) in enumerate(active_segments):
                log_content.append(f"  Segment {i+1}: {self.format_time(start)} - {self.format_time(end)} (duration: {end-start:.2f}s)")
            
            # Check if input has audio to determine command structure
            video_info = self.get_video_info(input_file)
            has_audio = video_info and video_info.get('has_audio', True)
            video_codec = video_info.get('video_codec', 'unknown') if video_info else 'unknown'
            
            # Determine best encoding strategy based on input codec and dimensions
            def get_video_encoding_params(input_codec, width, height):
                """Get appropriate video encoding parameters based on input codec and dimensions"""
                # Check if dimensions are even (required for most encoders)
                width_even = width % 2 == 0
                height_even = height % 2 == 0
                
                if input_codec in ['vp9', 'vp8']:
                    if width_even and height_even:
                        # Dimensions are good, can encode
                        return ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23']
                    else:
                        # Odd dimensions - VP9 with seeking issues, use padding approach
                        log_content.append(f"Warning: Odd dimensions ({width}x{height}) detected for VP9/VP8")
                        log_content.append("VP9 videos with odd dimensions may have seeking issues")
                        log_content.append("Using padding approach to avoid encoding problems")
                        new_width = width + (width % 2)
                        new_height = height + (height % 2)
                        return [
                            '-vf', f'pad={new_width}:{new_height}:(ow-iw)/2:(oh-ih)/2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23'
                        ]
                elif input_codec in ['h264', 'avc']:
                    # For H.264, we can usually copy or re-encode safely
                    return ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
                else:
                    if width_even and height_even:
                        # For other codecs, use H.264 with safe settings
                        return ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23']
                    else:
                        # Odd dimensions - use padding filter
                        new_width = width + (width % 2)
                        new_height = height + (height % 2)
                        log_content.append(f"Padding video from {width}x{height} to {new_width}x{new_height}")
                        return [
                            '-vf', f'pad={new_width}:{new_height}:(ow-iw)/2:(oh-ih)/2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23'
                        ]
            
            width = video_info.get('width', 1920) if video_info else 1920
            height = video_info.get('height', 1080) if video_info else 1080
            video_params = get_video_encoding_params(video_codec, width, height)
            
            # For single segment, use simple approach
            if len(active_segments) == 1:
                start, end = active_segments[0]
                
                base_cmd = [
                    'ffmpeg', '-i', input_file,
                    '-ss', str(start), '-t', str(end - start)
                ]
                
                # Check if we're using stream copy (which handles audio automatically)
                using_stream_copy = '-c' in video_params and 'copy' in video_params
                
                if using_stream_copy:
                    # Stream copy preserves everything as-is
                    cmd = base_cmd + video_params + [
                        '-avoid_negative_ts', 'make_zero',
                        '-y', output_file
                    ]
                elif has_audio:
                    cmd = base_cmd + video_params + [
                        '-c:a', 'aac', '-b:a', '128k',
                        '-avoid_negative_ts', 'make_zero',
                        '-y', output_file
                    ]
                else:
                    cmd = base_cmd + video_params + [
                        '-an',  # No audio
                        '-avoid_negative_ts', 'make_zero',
                        '-y', output_file
                    ]
                
                log_content.append("Using single segment extraction method")
                log_content.append(f"Input codec: {video_codec}")
                log_content.append(f"Audio track: {'Yes' if has_audio else 'No'}")
                log_content.append(f"FFmpeg command: {' '.join(cmd)}")
                
            else:
                # For multiple segments, use concat demuxer approach (more reliable)
                temp_dir = tempfile.mkdtemp()
                segment_files = []
                concat_list_file = os.path.join(temp_dir, 'concat_list.txt')
                
                try:
                    # Extract each segment to a temporary file
                    for i, (start, end) in enumerate(active_segments):
                        segment_file = os.path.join(temp_dir, f'segment_{i:03d}.mp4')
                        segment_files.append(segment_file)
                        
                        base_cmd = [
                            'ffmpeg', '-i', input_file,
                            '-ss', str(start), '-t', str(end - start)
                        ]
                        
                        # Check if we're using stream copy
                        using_stream_copy = '-c' in video_params and 'copy' in video_params
                        
                        if using_stream_copy:
                            # Stream copy preserves everything as-is
                            cmd = base_cmd + video_params + [
                                '-avoid_negative_ts', 'make_zero',
                                '-y', segment_file
                            ]
                        elif has_audio:
                            cmd = base_cmd + video_params + [
                                '-c:a', 'aac', '-b:a', '128k',
                                '-avoid_negative_ts', 'make_zero',
                                '-y', segment_file
                            ]
                        else:
                            cmd = base_cmd + video_params + [
                                '-an',  # No audio
                                '-avoid_negative_ts', 'make_zero',
                                '-y', segment_file
                            ]
                        
                        log_content.append(f"Extracting segment {i+1}/{len(active_segments)}: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            log_content.append(f"Failed to extract segment {i+1}")
                            log_content.append(f"Error: {result.stderr}")
                            log_content.append(f"Command was: {' '.join(cmd)}")
                            return False
                        else:
                            # Verify segment was created and has reasonable size
                            if os.path.exists(segment_file):
                                size = os.path.getsize(segment_file)
                                log_content.append(f"Segment {i+1} created successfully ({size} bytes)")
                                
                                # Check if segment is too small (indicates stream copy failure)
                                if size < 1024 and using_stream_copy:
                                    log_content.append(f"WARNING: Segment {i+1} is very small ({size} bytes)")
                                    log_content.append("Stream copy may have failed - will trigger fallback processing")
                                    # Don't return False here, let the concatenation fail and trigger fallback
                            else:
                                log_content.append(f"ERROR: Segment {i+1} file was not created")
                                return False
                    
                    # Create concat list file
                    with open(concat_list_file, 'w') as f:
                        for segment_file in segment_files:
                            f.write(f"file '{segment_file}'\n")
                    
                    # Concatenate all segments
                    cmd = [
                        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_file,
                        '-c', 'copy', '-y', output_file
                    ]
                    
                    log_content.append("Using multi-segment concatenation method")
                    log_content.append(f"Temporary directory: {temp_dir}")
                    
                finally:
                    # Clean up temporary files
                    try:
                        for segment_file in segment_files:
                            if os.path.exists(segment_file):
                                os.unlink(segment_file)
                        if os.path.exists(concat_list_file):
                            os.unlink(concat_list_file)
                        os.rmdir(temp_dir)
                    except Exception as cleanup_error:
                        log_content.append(f"Warning: Failed to clean up temp files: {cleanup_error}")
            
            # Execute the final command
            log_content.append(f"Final FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                log_content.append("")
                log_content.append("Video processing completed successfully")
                
                # Verify output file was created and has reasonable size
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    log_content.append(f"Output file size: {file_size / (1024*1024):.2f} MB")
                    if file_size < 1024:  # Less than 1KB is suspicious
                        log_content.append("Warning: Output file is very small, may be corrupted")
                        return False
                else:
                    log_content.append("Error: Output file was not created")
                    return False
                    
                return True
            else:
                log_content.append(f"FFmpeg error (return code {result.returncode}):")
                log_content.append(f"STDERR: {result.stderr}")
                log_content.append(f"STDOUT: {result.stdout}")
                
                # Try fallback method for problematic codecs like VP9
                if video_codec in ['vp9', 'vp8'] and len(active_segments) > 1:
                    log_content.append("")
                    log_content.append("Trying fallback method for VP9/VP8 codec...")
                    return self.fallback_video_processing(input_file, output_file, active_segments, log_content)
                
                return False
                
        except Exception as e:
            log_content.append(f"Exception in create_processed_video: {str(e)}")
            import traceback
            log_content.append(f"Traceback: {traceback.format_exc()}")
            return False
    
    def fallback_video_processing(self, input_file, output_file, active_segments, log_content):
        """Fallback method for problematic video formats like VP9"""
        try:
            log_content.append("Using fallback processing method...")
            
            if len(active_segments) == 1:
                # Single segment - try stream copy first
                log_content.append("Single segment - trying stream copy...")
                start, end = active_segments[0]
                cmd = [
                    'ffmpeg', '-i', input_file,
                    '-ss', str(start), '-t', str(end - start),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y', output_file
                ]
                
                log_content.append(f"Fallback command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_content.append("Stream copy method succeeded!")
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        log_content.append(f"Output file size: {file_size / (1024*1024):.2f} MB")
                        return file_size > 1024
                    return False
                else:
                    log_content.append(f"Stream copy failed: {result.stderr}")
                    return False
            else:
                # Multiple segments - try a different approach
                log_content.append("Multiple segments - trying VP9 preservation with padding...")
                
                # For VP9 with odd dimensions, add padding to make dimensions even
                filter_parts = []
                input_specs = []
                
                for i, (start, end) in enumerate(active_segments):
                    input_specs.extend(['-ss', str(start), '-t', str(end - start), '-i', input_file])
                    # Add padding filter to make dimensions even
                    filter_parts.append(f'[{i}:v]pad=1030:666:(ow-iw)/2:(oh-ih)/2[v{i}]')
                
                # Concatenate the padded videos
                concat_inputs = ''.join(f'[v{i}]' for i in range(len(active_segments)))
                full_filter = ';'.join(filter_parts) + f';{concat_inputs}concat=n={len(active_segments)}:v=1[outv]'
                
                cmd = ['ffmpeg'] + input_specs + [
                    '-filter_complex', full_filter,
                    '-map', '[outv]',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-crf', '28',  # Higher CRF for faster encoding
                    '-y', output_file
                ]
                
                log_content.append(f"Fallback command with padding: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_content.append("Fallback method with padding succeeded!")
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        log_content.append(f"Output file size: {file_size / (1024*1024):.2f} MB")
                        return file_size > 1024
                    return False
                else:
                    log_content.append(f"Fallback method with padding failed: {result.stderr}")
                    
                    # Last resort - create a simple concat with re-encoding each segment individually
                    log_content.append("Trying last resort: individual segment processing...")
                    return self.last_resort_processing(input_file, output_file, active_segments, log_content)
                
        except Exception as e:
            log_content.append(f"Exception in fallback processing: {str(e)}")
            return False
    
    def last_resort_processing(self, input_file, output_file, active_segments, log_content):
        """Last resort processing for very problematic videos"""
        try:
            temp_dir = tempfile.mkdtemp()
            segment_files = []
            
            try:
                # Process each segment individually with padding
                for i, (start, end) in enumerate(active_segments):
                    segment_file = os.path.join(temp_dir, f'segment_{i:03d}.mp4')
                    segment_files.append(segment_file)
                    
                    # Extract and pad each segment
                    cmd = [
                        'ffmpeg', '-i', input_file,
                        '-ss', str(start), '-t', str(end - start),
                        '-vf', 'pad=1030:666:(ow-iw)/2:(oh-ih)/2',  # Pad to even dimensions
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast', '-crf', '28',
                        '-an',  # No audio for simplicity
                        '-y', segment_file
                    ]
                    
                    log_content.append(f"Processing segment {i+1} individually...")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        log_content.append(f"Failed to process segment {i+1}: {result.stderr}")
                        return False
                    
                    if not os.path.exists(segment_file) or os.path.getsize(segment_file) < 1024:
                        log_content.append(f"Segment {i+1} is invalid or too small")
                        return False
                
                # Create concat list file
                concat_list_file = os.path.join(temp_dir, 'concat_list.txt')
                with open(concat_list_file, 'w') as f:
                    for segment_file in segment_files:
                        f.write(f"file '{segment_file}'\n")
                
                # Concatenate all segments
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_file,
                    '-c', 'copy', '-y', output_file
                ]
                
                log_content.append("Final concatenation...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_content.append("Last resort processing succeeded!")
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        log_content.append(f"Output file size: {file_size / (1024*1024):.2f} MB")
                        return file_size > 1024
                    return False
                else:
                    log_content.append(f"Final concatenation failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temporary files
                try:
                    for segment_file in segment_files:
                        if os.path.exists(segment_file):
                            os.unlink(segment_file)
                    if os.path.exists(concat_list_file):
                        os.unlink(concat_list_file)
                    os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    log_content.append(f"Warning: Failed to clean up temp files: {cleanup_error}")
            
        except Exception as e:
            log_content.append(f"Exception in last resort processing: {str(e)}")
            return False

def main():
    root = tk.Tk()
    app = VideoCallProcessor(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()