
# Player Re-Identification in Sports Videos

This project performs player re-identification in a single camera feed using a fine-tuned YOLO model. It assigns consistent IDs to players even if they temporarily leave and re-enter the frame.

## Features

- Real-time player detection using a custom-trained YOLO model
- Consistent ID assignment based on spatial tracking
- Simple and lightweight implementation using OpenCV
- Customizable for any sports video with similar camera angles

## Project Structure

```
player_tracking_project/
├── best.pt                 # Custom YOLO model trained on players
├── 15sec_input_720p.mp4    # Input video
├── track_players.py        # Main tracking script
├── reid_output.avi         # Output video (generated)
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Create a Python Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

### 2. Install Dependencies

Install required Python libraries:

```bash
pip install ultralytics opencv-python
```

(Optional, for headless environments or video export issues):

```bash
pip install opencv-python-headless
```

### 3. Prepare Files

- Place your custom YOLO model (`best.pt`) in the root directory.
- Place your input video (`15sec_input_720p.mp4`) in the same folder.
- Ensure the file paths inside `track_players.py` match your filenames.

### 4. Run the Script

```bash
python track_players.py
```

The script will process the video, detect players, assign IDs, and save the output video as `reid_output.avi`.

## Notes

- This implementation uses basic centroid tracking with distance-based ID matching.
- To improve tracking accuracy or handle occlusions, consider integrating Deep SORT or ByteTrack.
- The model assumes that class ID 0 corresponds to "player".

## License

This project is open-source and available under the MIT License.
