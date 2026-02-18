import random
from nltk.corpus import wordnet

def replace_multiple_words(sentence, n=2):
    words = sentence.split()
    candidates = [w for w in words if wordnet.synsets(w)]
    if not candidates:
        return sentence

    words_to_replace = random.sample(candidates, min(n, len(candidates)))
    new_words = words.copy()

    for target in words_to_replace:
        synsets = wordnet.synsets(target)
        lemmas = set(lemma.name().replace("_", " ") for s in synsets for lemma in s.lemmas())
        synonyms = list(lemmas - {target})
        if synonyms:
            new_word = random.choice(synonyms)
            new_words = [new_word if w == target else w for w in new_words]

    return " ".join(new_words)

# Example
sentence = "the smart student studies hard in the library every night for weeks before his exam"
print(replace_multiple_words(sentence, n=3))
