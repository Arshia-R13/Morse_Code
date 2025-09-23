
import cv2
import numpy as np
import mediapipe as mp
import time

from typing import List, Tuple
from functools import lru_cache
import heapq
import itertools



##############################################################
###############     Video-Datei einlesen:    #################
video_path = "Videos/Elham01.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Arshia00.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Arshia01.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Help.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Kasra.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Sadra.mp4"  # <-- Pfad zu deinem Video

###### meine Videos:
#video_path = "Videos/Hello01.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Hello02.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Hallo.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Hilfe.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Love.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Move.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Arshia.mp4"  # <-- Pfad zu deinem Video
#video_path = "Videos/Elham.mp4"  # <-- Pfad zu deinem Video

##############################################################

cap = cv2.VideoCapture(video_path)

# Ausgabe
print(f"Frame rate: {cap.get(cv2.CAP_PROP_FPS)}\n")

