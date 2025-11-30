# Subtitle generation for YouTube Videos

This research aims to enhance video subtitle alignment
and segmentation for better accessibility and
viewing experiences. Key objectives include:

1. Utilizing fine tuned Whisper models for translating speech to text.

2. Implementing text segmentation techniques
with state-of-the-art language models for generating refined subtitles of reasonable length.

3. Creating a robust methodology to validate caption
quality across content types.

4. Aligning subtitles appropriately to speech without altering
original video timestamps.


# Streamlit GUI

The [app_transcribe.py](https://github.com/sing1174/Subtitle_generation/blob/main/app_transcribe.py)  file includes code for implemeting the GUI. 

Use the following command to run the app: 
`streamlit run app_transcribe.py`.

Here is a snapshot of the interface, downloading audio and
video from a YouTube link given as input and generating
Improved SRT file for Subtitles:

![Display of GUI, downloading audio and
video from a YouTube link given as input and generating
Improved SRT file for Subtitles](https://github.com/anwesha-umn/TranscribeAI_subtitle_generation/blob/main/images/gui1.png)

Custom segmentation helps to refine the captions,
ensuring accurate and well-structured subtitles.
The refined SRT files are embedded into the
video using FFmpeg, creating a captioned video
output. With options to preview the video and
download the SRT file, the pipeline offers a complete
solution for showcasing the workflow and
results in an interactive demo.


The final output video with added Subtitles:
![gui2](https://github.com/anwesha-umn/TranscribeAI_subtitle_generation/blob/main/images/gui2.png)

