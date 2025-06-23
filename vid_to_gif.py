!pip install moviepy

from moviepy.editor import VideoFileClip

videoClip = VideoFileClip(".mp4")

videoClip.write_gif(".gif")
