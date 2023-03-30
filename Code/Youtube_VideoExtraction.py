import os
import pytube
from pytube import Playlist

link = "Enter URL"
yt_playlist = Playlist(link)
for video in yt_playlist.videos:
    video.streams.get_highest_resolution().download("Path")
    print("Video Downloaded: ", video.title)

print("\n All videos downloaded")