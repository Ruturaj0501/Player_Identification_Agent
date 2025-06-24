# Player_Identification_Agent
1. Create Your Project Folder
Make a folder called something like player_tracking_project. Inside it, place:
Your YOLO model → best.pt
Your input video → 15sec_input_720p.mp4
Your Python script → track_players.py (copy-paste the main code into this file)


2. Install Python & Set Up Environment
If you don’t already have Python:
Install it from https://www.python.org (recommended version: 3.10 or 3.11)
Then open Command Prompt or Terminal and run:
python -m venv venv

Then activate the environment:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate


3. Install Required Libraries
Once your environment is active, install the required Python packages:
pip install ultralytics opencv-python

If you also want to save videos or run in servers without display (optional):
pip install opencv-python-headless


4. Edit Your Script (track_players.py)
Inside the script:

Make sure the paths to your files are correct:
python
video_path = "15sec_input_720p.mp4"
model_path = "best.pt"
output_path = "reid_output.avi"


5. Run the Script
Just run this in the terminal while inside the folder:
python track_players.py


6. See Results
After the code finishes:
You’ll get a file named reid_output.avi in the same folder.
Open it to watch your tracked players, each with a consistent ID!
