import string
from pathlib import Path
from collections import Counter
from itertools import chain
import operator

ALLOWED_ATTEMPTS = 6
WORD_LENGTH = 5

ALLOWABLE_CHARACTERS = set(string.ascii_lowercase)
DICT = "data/words_alpha.txt"

# get word with required length and characters
WORDS = {
  word.lower()
  for word in Path(DICT).read_text().splitlines()
  if len(word) == WORD_LENGTH and set(word) < ALLOWABLE_CHARACTERS
}

# occurence of letters in words
LETTER_COUNTER = Counter(chain.from_iterable(WORDS))

# probability of letters in words
LETTER_FREQUENCY = {
    character: value / sum(LETTER_COUNTER.values())
    for character, value in LETTER_COUNTER.items()
}

def calculate_word_commonality(word):
    score = 0.0
    for char in word:
        score += LETTER_FREQUENCY[char]
    return score / (WORD_LENGTH - len(set(word)) + 1)

def sort_by_word_commonality(words):
    sort_by = operator.itemgetter(1)
    return sorted(
        [(word, calculate_word_commonality(word)) for word in words],
        key=sort_by,
        reverse=True,
    )

def input_word():
    while True:
        word = input("Input the word you entered> ")
        if len(word) == WORD_LENGTH and word.lower() in WORDS:
            break
    return word.lower()

# G = all correct Y = position incorrect ? = all incorrect
def get_response(input, answer):
    response = ""
    for idx, letter in enumerate(input):
        if letter == answer[idx]:
            response += "G"
        elif letter in answer:
            response += "Y"
        else:
            response += "?"
    return response

word_vector = [set(string.ascii_lowercase) for _ in range(WORD_LENGTH)]

def match_word_vector(word, word_vector):
    assert len(word) == len(word_vector)
    for letter, v_letter in zip(word, word_vector):
        if letter not in v_letter:
            return False
    return True

def match(word_vector, possible_words):
    return [word for word in possible_words if match_word_vector(word, word_vector)]

def solver(answer):
    words = []
    responses = []
    possible_words = WORDS.copy()
    word_vector = [set(string.ascii_lowercase) for _ in range(WORD_LENGTH)]
    for attempt in range(1, ALLOWED_ATTEMPTS + 1):
        word = sort_by_word_commonality(possible_words)[0][0]
        words.append(word)
        response = get_response(word, answer)
        responses.append(response)
        for idx, letter in enumerate(response):
            if letter == "G":
                word_vector[idx] = {word[idx]}
            elif letter == "Y":
                try:
                    word_vector[idx].remove(word[idx])
                except KeyError:
                    pass
            elif letter == "?":
                for vector in word_vector:
                    try:
                        vector.remove(word[idx])
                    except KeyError:
                        pass
        possible_words = match(word_vector, possible_words)
        if word == answer:
          return words
    return words

def step_solver(answer, possible_words):
    word_vector = [set(string.ascii_lowercase) for _ in range(WORD_LENGTH)]
    word = sort_by_word_commonality(possible_words)[0][0]
    response = get_response(word, answer)
    for idx, letter in enumerate(response):
        if letter == "G":
            word_vector[idx] = {word[idx]}
        elif letter == "Y":
            try:
                word_vector[idx].remove(word[idx])
            except KeyError:
                pass
        elif letter == "?":
            for vector in word_vector:
                try:
                    vector.remove(word[idx])
                except KeyError:
                    pass
    possible_words = match(word_vector, possible_words)
    
    return word, possible_words

# # sample usage
# answer = 'hidel'
# words, responses = solver(answer)

# print(words)
# print(responses)