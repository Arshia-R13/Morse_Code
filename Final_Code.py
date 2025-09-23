
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

# Ausgabe:
print(f"Frame rate: {cap.get(cv2.CAP_PROP_FPS)}\n")


frames = []
sequence = []      # Liste zur Speicherung von 1 und 0
prev_state = None  # Vorheriger Zustand
last_switch_time = 0
interval = 3       # Sekunden zwischen Zustandsänderungen
frame_rate = cap.get(cv2.CAP_PROP_FPS)


frame_count = 0
hand_lost_counter = 0

def is_hand_open(hand_landmarks):
    """Einfacher Heuristik-Ansatz: Ist die Hand offen?"""
    tips_ids = [8, 12, 16, 20]  # Fingerspitzen
    for tip_id in tips_ids:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            return False  # Mindestens ein Finger nicht ausgestreckt
    return True

# Ausgabe:
print(f"Video: {video_path} wird analysiert...\n")

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
                if frame_count >= STABLE_MIN_FRAMES and current_state != -1:
                    final_sequence.append(current_state)

                current_state = None
                frame_count = 0
                gap_count = 0
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

# Ausgabe
print("Erkannte Binärfolge (1 = offen, 0 = geschlossen, -1 = Pause):")
print(f"\nvideo_path:{video_path}")
print("Finale Bitfolge:", final_sequence)


########## Teil 2:
### Decoder:

# --- Hilfsfunktionen ---
# Hf1:
def hamming_distance(a, b):
    """Berechnet Hamming-Distanz zwischen zwei Bitfolgen."""
    if len(a) != len(b):
        # wenn unterschiedlich lang: Strafe
        min_len = min(len(a), len(b))
        dist = sum(x != y for x, y in zip(a[:min_len], b[:min_len]))
        dist += abs(len(a) - len(b))
        return dist
    return sum(x != y for x, y in zip(a, b))

# Hf2:
def bits_for_word(word, morse_table):
    """Wandle ein Wort oder Phrase in eine Bitfolge um."""
    seq = []
    for ch in word.replace(" ", ""):   # Leerzeichen ignorieren
        if ch in morse_table:
            seq.extend(morse_table[ch])
    return seq


# Hf3: Dynamischer Lexikon-Kombinator
def expand_lexicon(base_words, morse_table, max_len=4):
    """
    Erzeugt neue Phrasen (z.B. 'MEIN NAME IST ARSHIA') automatisch.
    max_len = maximale Anzahl Wörter pro Phrase
    """
    lexicon = set(base_words)

    # Alle möglichen Kombinationen bis max_len
    for l in range(2, max_len + 1):
        for combo in itertools.permutations(base_words, l):
            phrase = " ".join(combo)
            lexicon.add(phrase)

    return lexicon


# Hf4: Beam Search
def decode_bits_beam(bits, morse_table, lexicon, beam_size=10):
    """Sucht das nächstliegende Wort oder Phrase im Lexikon."""
    results = []

    for word in lexicon:
        word_bits = bits_for_word(word, morse_table)
        dist = hamming_distance(bits, word_bits)
        heapq.heappush(results, (dist, word))

    best = heapq.nsmallest(beam_size, results)
    return [(w, d) for d, w in best]


# --- Hauptfunktion ---
def decode_with_lexicon_or_estimate(bits, morse_table, lexicon, max_hamming=1):
    """
    Zuerst harte Lexikonprüfung, dann offene Schätzung.
    bits: Liste von 0/1
    morse_table: dict {Buchstabe: Bitfolge}
    lexicon: Menge an gültigen Wörtern
    """
    # Lexikon-Harte Suche
    best_lex_match = None
    best_lex_dist = float('inf')

    for word in lexicon:
        # Morse-Bitfolge für das Wort erzeugen
        word_bits = bits_for_word(word, morse_table)
        dist = hamming_distance(bits, word_bits)
        if dist <= max_hamming and dist < best_lex_dist:
            best_lex_match = word
            best_lex_dist = dist

    if best_lex_match:
        return best_lex_match  # Sofortiger Rückgabewert bei Lexikon-Treffer

    # Offene Schätzung via Beam Search
    best_candidates = decode_bits_beam(bits, morse_table, lexicon, beam_size=10)
    return best_candidates[0][0]  # nur das beste Ergebnis zurückgeben


MORSE_TABLE_STR = {
    "A": "01",     "B": "1000",  "C": "1010", "D": "100",  "E": "0",
    "F": "0010",   "G": "110",   "H": "0000", "I": "00",   "J": "0111",
    "K": "101",    "L": "0100",  "M": "11",   "N": "10",   "O": "111",
    "P": "0110",   "Q": "1101",  "R": "010",  "S": "000",  "T": "1",
    "U": "001",    "V": "0001",  "W": "011",  "X": "1001", "Y": "1011",
    "Z": "1100",   " ": ""
}

# Umwandlung: String -> Liste von ints
MORSE_TABLE = {letter: [int(ch) for ch in code]
               for letter, code in MORSE_TABLE_STR.items()}

# --- Wörter ---
BASE_WORDS = {"HALLO", "HELLO", "WELT", "TEST", "HILFE", "MORSE", "CODE",
              "ICH", "MEIN", "NAME", "IST", "BRAUCHE", "ARSHIA", "ELHAM",
              "LOVE", "MOVE", "HELP", "SADRA", "KASRA"}

# Lexikon erweitern (mit dynamischen Kombinationen bis 4 Wörter)
LEXICON = expand_lexicon(BASE_WORDS, MORSE_TABLE, max_len=4)

bitfolge = final_sequence

text = decode_with_lexicon_or_estimate(
    bitfolge,
    MORSE_TABLE,
    LEXICON,
    max_hamming=2
)

print(f"\nBeste Wort-Dekodierung: '{text}' oder: ", text.lower())





##########################################
########         Ausgaben    #############
##########################################
# 1:
# video_path:Elham01.mp4
# Finale Bitfolge: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
# Beste Wort-Dekodierung: 'ELHAM' oder:  elham
#########
# 2:
# video_path:Arshia00.mp4
# Finale Bitfolge: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# Beste Wort-Dekodierung: 'ARSHIA' oder:  arshia
#########
# 3:
# video_path:Arshia01.mp4
# Finale Bitfolge: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# Beste Wort-Dekodierung: 'ARSHIA' oder:  arshia
#########
# 4:
# video_path:Help.mp4
# Finale Bitfolge: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
# Beste Wort-Dekodierung: 'HELP' oder:  help
#########
# 5:
# video_path:Kasra.mp4
# Finale Bitfolge: [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
# Beste Wort-Dekodierung: 'KASRA' oder:  kasra
#########
# 6:
# video_path:Sadra.mp4
# Finale Bitfolge: [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
# Beste Wort-Dekodierung: 'SADRA' oder:  sadra
#########
# 7)
# video_path:Hello01.mp4
# Finale Bitfolge: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
# Beste Wort-Dekodierung: 'HELLO' oder:  hello
#########
# 8)
# video_path:Hello02.mp4
# Finale Bitfolge: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]
# Beste Wort-Dekodierung: 'HALLO' oder:  hallo
#########
# 9)
# video_path:Hello03.mp4
# Finale Bitfolge: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
#########
# 10)
# video_path:Hilfe.mp4
# Finale Bitfolge: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
# Beste Wort-Dekodierung: 'HILFE' oder:  hilfe
#########
# 11)
# video_path:Love.mp4
# Finale Bitfolge: [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
# Beste Wort-Dekodierung: 'LOVE' oder:  love
#########
# 12)
# video_path:Move.mp4
# Finale Bitfolge: [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
# Beste Wort-Dekodierung: 'MOVE' oder:  move
#########
# 13)
# video_path:Arshia.mp4
# Finale Bitfolge: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# Beste Wort-Dekodierung: 'ARSHIA' oder:  arshia
#########
# 14)
# video_path:Elham.mp4
# Finale Bitfolge: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
# Beste Wort-Dekodierung: 'ELHAM' oder:  elham
#########


