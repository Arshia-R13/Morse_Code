
import cv2
import numpy as np
import mediapipe as mp
import time

from typing import List, Tuple
from functools import lru_cache
import heapq
import itertools


# --- Hilfsfunktionen ---
# 1:
def hamming_distance(a, b):
    """Berechnet Hamming-Distanz zwischen zwei Bitfolgen."""
    if len(a) != len(b):
        # wenn unterschiedlich lang: Strafe
        min_len = min(len(a), len(b))
        dist = sum(x != y for x, y in zip(a[:min_len], b[:min_len]))
        dist += abs(len(a) - len(b))
        return dist
    return sum(x != y for x, y in zip(a, b))

# 2:
def bits_for_word(word, morse_table):
    """Wandle ein Wort oder Phrase in eine Bitfolge um."""
    seq = []
    for ch in word.replace(" ", ""):   # Leerzeichen ignorieren
        if ch in morse_table:
            seq.extend(morse_table[ch])
    return seq


# 3: Dynamischer Lexikon-Kombinator
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


# 4: Beam Search
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

video_path = "Videos/Hallo.mp4"  # <-- Pfad zu deinem Video
print(f"\nvideo_path:{video_path}")

print(f"Video: {video_path} wird analysiert...\n")
finale_Bitfolge = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]

final_sequence = finale_Bitfolge
print("Finale Bitfolge:", final_sequence)

bitfolge = final_sequence

text = decode_with_lexicon_or_estimate(
    bitfolge,
    MORSE_TABLE,
    LEXICON,
    max_hamming=2
)

print(f"\nBeste Wort-Dekodierung: '{text}' oder '{text.lower()}'")
