
letter_to_int_dict = {
    'padding': 0,
    'a': 1,
    'b': 2, 
    'c': 3, 
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'z': 22,
    'x': 23,
    'y': 24,
    'w': 25,
    'q': 26,
    '-': 27,
    ' ': 28,
    '.': 29,
    ',': 30,
    '!': 31,
    '?': 32,
    '_': 33,
    '$': 34,
    '&': 35,
    ')': 36,
    '(': 37,
    '+': 38,
    '"': 39,
    "'": 40,
    'end': 41
}

def map_letter_to_int(value):
    """
    Maps given letter or string to integer. For non-important characters it will return 42.
    :param value: String or character to map. Possbile strings: 'end' and 'padding'.
    :return: Integer value for given string or character.
    """
    return letter_to_int_dict.get(value, 42) # 42 is used for non-important characters

