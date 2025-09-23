
import cv2
import numpy as np
import mediapipe as mp
import time

from typing import List, Tuple
from functools import lru_cache
import heapq
import itertools

# MediaPipe initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

video_path = "Videos/Hallo.mp4"  # <-- Pfad zu deinem Video
cap = cv2.VideoCapture(video_path)


frames = []
sequence = []      # Liste zur Speicherung von 1 und 0
prev_state = None  # Vorheriger Zustand
last_switch_time = 0
interval = 3       # Sekunden zwischen Zustandsänderungen
frame_rate = cap.get(cv2.CAP_PROP_FPS)


frame_count = 0
hand_lost_counter = 0

# Ausgabe:
print(f"Video: {video_path} wird analysiert...\n")

def is_hand_open(hand_landmarks):
    """Einfacher Heuristik-Ansatz: Ist die Hand offen?"""
    tips_ids = [8, 12, 16, 20]  # Fingerspitzen
    for tip_id in tips_ids:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            return False  # Mindestens ein Finger nicht ausgestreckt
    return True



while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    current_time = frame_count / frame_rate

    if results.multi_hand_landmarks:
        hand_lost_counter = 0  # Hand ist sichtbar
        hand_landmarks = results.multi_hand_landmarks[0]
        state = 1 if is_hand_open(hand_landmarks) else 0


        # Wenn Zustand neu ist oder Zeit vergangen ist

        sequence.append(state)

    else:
        sequence.append(-1)  # -1 als Trenner für neue Zeile/Sequenz

cap.release()
hands.close()


frames = sequence

STABLE_MIN_FRAMES = 31     # Mindestlänge für stabilen Zustand
GAP_TOLERANCE = 20         # Maximal erlaubte Unterbrechung in Frames, bevor neuer Zustand gezählt wird

final_sequence = []
current_state = None
frame_count = 0
gap_count = 0

for value in frames:  # frames = deine Frame-Analyse
    if current_state is None:
        current_state = value
        frame_count = 1
        gap_count = 0
    else:
        if value == current_state:
            frame_count += 1
            gap_count = 0  # Unterbrechung beendet
        elif value == -1:
            # Kurze Unterbrechung innerhalb eines Zustands ignorieren
            gap_count += 1
            if gap_count > GAP_TOLERANCE:
                # Zustand beenden
                if (frame_count >= STABLE_MIN_FRAMES and current_state != -1):
                    final_sequence.append(current_state)

                frame_count = 0
                gap_count = 0
                current_state = None
        else:
            # Wechsel zu anderem Wert
            if frame_count >= STABLE_MIN_FRAMES and current_state != -1:
                final_sequence.append(current_state)

                frame_count = 1
                gap_count = 0

            current_state = value



# Letzten Zustand speichern, falls lang genug
if frame_count >= STABLE_MIN_FRAMES and current_state != -1:
    final_sequence.append(current_state)
    frame_count = 1

print(f"\nvideo_path:{video_path}")
print("Finale Bitfolge:", final_sequence)