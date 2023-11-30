import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from more_itertools import peekable

def read_set(file):
    s = set()
    for line in file:
        trimmed = line.strip()
        if trimmed:
            s.add(trimmed)
    return s

def only_alpha(s):
    return "".join([i for i in s if i.isalpha() or i == " "])

blocked_pos = ["CD", "POS", ".", "(", ")"]

class IngredientCorpus:
    def __init__(self):
        self.tokens = set()
        self.bktree = {}
        self.lem = WordNetLemmatizer()

    def load_tokens(self, token_file):
        with open(token_file, "r") as file:
            for token in file.readlines():
                if token:
                    self.tokens.add(token.strip())

    def load_bktree(self, tree_file):
        with open(tree_file, "r") as file:
            self.bktree = json.load(file)

    def _pre_token_strip(self, s):
        return " ".join(s.split()).lower()

    def _tokenize_ingr(self, txt):
        txt = self._pre_token_strip(txt)
        tokens = word_tokenize(txt)
        lemmatized_tokens = []
        for token, pos in nltk.pos_tag(tokens):
            token = only_alpha(token)
            if pos.startswith("N"):
                token = self.lem.lemmatize(token, "n")
            if token in self.tokens:
                lemmatized_tokens.append(token)
        return lemmatized_tokens

    def _find_matches(self, tokens):
        matches = set()
        peeker = peekable(tokens)
        while peeker:
            first_token = next(peeker)
            if first_token not in self.bktree:
                continue

            node = self.bktree[first_token]
            active_match = [first_token]
            def check_node():
                if "valid_end" in node:
                    ingr = " ".join(active_match)
                    matches.add(ingr)
            
            check_node()
            while peeker and peeker.peek() in node:
                next_lbl = next(peeker)
                node = node[next_lbl]
                active_match.append(next_lbl)
                check_node()
        return matches
    
    def _greedy_match(self, tokens):
        matches = list(self._find_matches(tokens))
        if len(matches) == 0:
            return None
        matches.sort(reverse=True, key=len)
        return matches[0]
    
    def find_match(self, ingredient):
        tokens =  self._tokenize_ingr(ingredient)
        return self._greedy_match(tokens)
