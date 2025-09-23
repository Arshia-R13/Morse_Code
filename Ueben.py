
import heapq
import itertools


# --- Hilfsfunktionen ---

def hamming_distance(a, b):
    """Berechnet Hamming-Distanz zwischen zwei Bitfolgen."""
    if len(a) != len(b):
        # wenn unterschiedlich lang: Strafe
        min_len = min(len(a), len(b))
        dist = sum(x != y for x, y in zip(a[:min_len], b[:min_len]))
        dist += abs(len(a) - len(b))
        return dist
    return sum(x != y for x, y in zip(a, b))


def bits_for_word(word, morse_table):
    """Wandle ein Wort oder Phrase in eine Bitfolge um."""
    seq = []
    for ch in word.replace(" ", ""):   # Leerzeichen ignorieren
        if ch in morse_table:
            seq.extend(morse_table[ch])
    return seq


# --- Dynamischer Lexikon-Kombinator ---
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


# --- Beam Search ---
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
        acc = 1 - best_lex_dist / max(len(bits), len(bits_for_word(best_lex_match, morse_table)))
        return best_lex_match, best_lex_dist, round(acc * 100, 2)  # Sofortiger Rückgabewert bei Lexikon-Treffer

    # Offene Schätzung via Beam Search
    best_candidates = decode_bits_beam(bits, morse_table, lexicon, beam_size=10)
    word, dist = best_candidates[0]
    acc = 1 - dist / max(len(bits), len(bits_for_word(word, morse_table)))
    return word, dist, round(acc * 100, 2)  # nur das beste Ergebnis zurückgeben


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
              "LOVE", "MOVE", "HELP", "SADRA", "KASRA"}     # "INFORMATIK"

# Lexikon erweitern (mit dynamischen Kombinationen bis 4 Wörter)
LEXICON = expand_lexicon(BASE_WORDS, MORSE_TABLE, max_len=4)

# --- Test ---
# 0:HELLO:
test0 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
#########
# 1:HALLO:
test1 = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
#########
# 2:HILFE:
test2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
#########
# 3:ARSHIA:
test3 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#########
# 4:ELHAM:
test4 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
#########
# 5:LOVE
test5 = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
#########
# 6:MOVE
test6 = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
#########
# 7:MEINNAME:
test7 = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
#########
# 8:MEINNAMEIST:
test8 = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
#########
# 9:ICHBRAUCHEHILFE:
test9 = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
#########
# 10:Help
test10 =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
#########
# 11:Kasra
test11 =  [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
#########
# 12:Sadra
test12 = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
#########
# 13:MEINNAMEISTARSHIA
test13 = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#########
# 14:Informatik
test14 = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]


tests = [globals()[f"test{i}"] for i in range(15)]

#[print(f'test{i}: {j}') for i,j in enumerate(tests)]

for i,t in enumerate(tests):
    print(f'\n##\ntest{i}: {t}')

    test_bits = t

    text, dist, acc = decode_with_lexicon_or_estimate(
        test_bits,
        MORSE_TABLE,
        LEXICON,
        max_hamming=2
    )
    print(f"Beste Wort-Dekodierung: '{text}' oder: ", text.lower())
    print(f"Distanz: {dist} | Genauigkeit: {acc}%")



"""
###########################################
###########    Ausgaben:    ###############
###########################################

##
0:HELLO:
erwartete = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
Ausgabe   = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
Beste Wort-Dekodierung: 'HELLO' oder:  hello
Distanz: 0 | Genauigkeit: 100.0%

##
1:HALLO:
erwartete = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
Ausgabe   = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
Beste Wort-Dekodierung: 'HALLO' oder:  'hallo'
Distanz: 0 | Genauigkeit: 100.0%

##
2:HILFE:
erwartete = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
Ausgabe   = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
Beste Wort-Dekodierung: 'HILFE' oder:  'hilfe'
Distanz: 0 | Genauigkeit: 100.0%

##
3:ARSHIA:
erwartete = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Ausgabe   = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Beste Wort-Dekodierung: 'ARSHIA' oder:  'arshia'
Distanz: 0 | Genauigkeit: 100.0%

##
4:ELHAM:
erwartete = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
Ausgabe   = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
Beste Wort-Dekodierung: 'ELHAM' oder:  'elham'
Distanz: 0 | Genauigkeit: 100.0%

##
5:LOVE
erwartete = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
Ausgabe   = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
Beste Wort-Dekodierung: 'LOVE' oder 'love'
Distanz: 0 | Genauigkeit: 100.0%

##
6:MOVE
erwartete = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
Ausgabe   = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
Beste Wort-Dekodierung: 'MOVE' oder:  'move'
Distanz: 0 | Genauigkeit: 100.0%

##
7:MEINNAME:
erwartete = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
Ausgabe   = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
Beste Wort-Dekodierung: 'MEIN NAME' oder 'mein name'
Distanz: 0 | Genauigkeit: 100.0%

##
8:MEINNAMEIST:
erwartete = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
Ausgabe   = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
Beste Wort-Dekodierung: 'MEIN NAME IST' oder 'mein name ist'
Distanz: 0 | Genauigkeit: 100.0%

##
9:ICHBRAUCHEHILFE:
erwartete = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
Ausgabe   = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
Beste Wort-Dekodierung: 'ICH BRAUCHE HILFE' oder 'ich brauche hilfe'
Distanz: 0 | Genauigkeit: 100.0%

##
10:Help
erwartete = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
Ausgabe   = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
Beste Wort-Dekodierung: 'HELP' oder:  'help'
Distanz: 0 | Genauigkeit: 100.0%

##
11:Kasra
erwartete = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
Ausgabe   = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
Beste Wort-Dekodierung: 'KASRA' oder:  'kasra'
Distanz: 0 | Genauigkeit: 100.0%

##
12:Sadra
erwartete = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
Ausgabe   = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
Beste Wort-Dekodierung: 'SADRA' oder:  'sadra'
Distanz: 0 | Genauigkeit: 100.0%

##
13:MEINNAMEISTARSHIA
erwartete = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Ausgabe   = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Beste Wort-Dekodierung: 'MEIN NAME IST ARSHIA' oder 'mein name ist arshia'
Distanz: 0 | Genauigkeit: 100.0%

##
14:Informatik
erwartete = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
Ausgabe   = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
Beste Wort-Dekodierung: 'ELHAM WELT' oder 'elham welt'
Distanz: 7 | Genauigkeit: 70.83%

"""