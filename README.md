# Video Call Recording Processor

A powerful tool for automatically detecting and removing inactive segments from video call recordings, making them shorter and more focused for analysis.

## Features

- **Universal Video Support**: Works with all common video formats (MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V, 3GP, OGV)
- **Intelligent Activity Detection**: 
  - Audio analysis to detect speech/sound activity
  - Ultra-sensitive visual analysis to detect even single pixel changes (perfect for typing detection)
  - Combined analysis for accurate activity detection
- **Automated Processing**: Removes inactive segments where there's no talking and no screen changes
- **Detailed Logging**: Generates comprehensive logs showing what was removed and why
- **Configurable Parameters**: Adjust sensitivity thresholds and minimum inactive duration
- **Batch Processing**: Process entire folders of video files at once

## How It Works

The tool analyzes video recordings in two ways:

1. **Audio Activity Detection**: Uses RMS energy analysis to detect when people are speaking
2. **Visual Activity Detection**: Compares consecutive frames to detect screen changes or movement
3. **Combined Analysis**: Merges both analyses to identify truly active segments
4. **Smart Filtering**: Only removes inactive segments longer than the specified minimum duration
5. **Video Generation**: Creates a new video file containing only the active segments

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg**:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Configure the activity detection settings:
   - **Audio Activity Threshold**: Lower values = more sensitive to quiet sounds (0.001-0.1)
   - **Visual Change Threshold**: Lower values = more sensitive to small changes (0.001-0.01) - detects single pixel changes like typing
   - **Min Inactive Duration**: Minimum seconds of inactivity required to remove a segment

3. Select a folder containing video files

4. Click "Process Video Files" to start batch processing

## Output

For each processed video file, the tool creates:

- **Processed Video**: `filename_processed.mp4` - Contains only active segments
- **Processing Log**: `logs/filename_log.txt` - Detailed analysis and removed segments

### Example Log Output

```
Processing Log for: meeting_recording.mp4
Started at: 2024-01-15 14:30:25
Settings: Audio Threshold=0.01, Visual Threshold=0.02, Min Inactive Duration=3.0s
--------------------------------------------------------------------------------
Video duration: 1800.00 seconds
Video FPS: 30.00

Audio analysis: Found 45 active audio segments
Total active audio time: 720.50 seconds (40.0%)
Visual analysis: Found 38 active visual segments
Total active visual time: 850.25 seconds (47.2%)

REMOVED INACTIVE SEGMENTS:
----------------------------------------
  00:02:15.000 - 00:04:30.000 (135.00s): No activity detected
  00:15:45.500 - 00:18:20.250 (154.75s): No activity detected
  00:25:10.000 - 00:28:05.000 (175.00s): No activity detected

SUMMARY:
  Original duration: 00:30:00.000 (1800.00s)
  Active segments: 12
  Total active time: 00:20:15.250 (1215.25s)
  Total removed time: 00:09:44.750 (584.75s)
  Time saved: 32.5%

Successfully created processed video: meeting_recording_processed.mp4
Completed at: 2024-01-15 14:35:42
```

## Configuration Tips

### For Different Types of Content

**Quiet Meetings/Calls**:
- Audio Threshold: 0.005-0.01
- Visual Threshold: 0.001-0.005 (detects typing and cursor movements)
- Min Inactive Duration: 2-3 seconds

**Noisy Environments**:
- Audio Threshold: 0.02-0.05
- Visual Threshold: 0.002-0.008 (still sensitive to typing)
- Min Inactive Duration: 5-10 seconds

**Screen Sharing Heavy (with typing/coding)**:
- Audio Threshold: 0.01-0.02
- Visual Threshold: 0.0005-0.002 (ultra-sensitive for single pixel changes)
- Min Inactive Duration: 3-5 seconds

**Presentation/Lecture**:
- Audio Threshold: 0.005-0.015
- Visual Threshold: 0.001-0.005 (catches slide changes and pointer movements)
- Min Inactive Duration: 3-5 seconds

## Technical Details

### Audio Analysis
- Extracts audio at 16kHz mono
- Uses RMS energy calculation with 0.5-second windows
- Detects speech activity above the configured threshold

### Visual Analysis
- Samples frames every 0.2 seconds for high sensitivity
- Uses higher resolution (640x480) to catch subtle changes like cursor movements
- Multiple detection methods:
  - Pixel change counting (detects any pixel that changes)
  - Average intensity change analysis
  - Maximum change detection (catches bright cursors, typing indicators)
- Ultra-sensitive to single pixel changes - perfect for detecting typing activity

### Video Processing
- Uses FFmpeg for all video operations
- Concatenates active segments seamlessly
- Maintains original video quality with H.264 encoding
- Preserves audio synchronization

## Requirements

- Python 3.7+
- FFmpeg (must be available in system PATH)
- OpenCV (opencv-python)
- Librosa (for audio analysis)
- SoundFile (for audio I/O)
- NumPy
- Tkinter (usually included with Python)

## Troubleshooting

**"FFmpeg is required but not found"**:
- Install FFmpeg and ensure it's in your system PATH
- Test by running `ffmpeg -version` in command line

**Audio analysis fails**:
- Check if the video file has an audio track
- Some formats may need conversion first

**Visual analysis slow**:
- Reduce visual threshold for faster processing
- Large video files will take longer to analyze

**Memory issues with large files**:
- Process files one at a time instead of batch processing
- Close other applications to free up RAM

## License

This tool is provided as-is for educational and productivity purposes.
