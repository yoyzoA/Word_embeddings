import random
import os
import argparse
from nltk.corpus import wordnet as wn
import nltk
#python eda_ptb.py --input ptb-sentences.txt --output ptb_augmented.txt --n_augments 2 --n_replacements 2 --replace_prob 0.2

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def get_synonyms(word, pos_tag=None):
    synonyms = set()

    # Map POS tag
    wn_pos = None
    if pos_tag:
        if pos_tag.startswith('J'):
            wn_pos = wn.ADJ
        elif pos_tag.startswith('V'):
            wn_pos = wn.VERB
        elif pos_tag.startswith('N'):
            wn_pos = wn.NOUN
        elif pos_tag.startswith('R'):
            wn_pos = wn.ADV

    # WordNet synonyms
    for syn in wn.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != word.lower():
                synonyms.add(lemma_name)

        # OMW synonyms (multilingual)
        for lemma in syn.lemmas():
            for lang in wn.langs():   # <-- OMW languages are inside wordnet now
                try:
                    for trans in lemma.synset().lemmas(lang):
                        name = trans.name().replace('_', ' ')
                        if name.lower() != word.lower():
                            synonyms.add(name)
                except:
                    pass

    return list(synonyms)

def synonym_replacement(sentence, n_replacements=1, replace_prob=0.1):
    """
    Replace up to n_replacements words in the sentence with synonyms.
    replace_prob = probability of each eligible word being replaced.
    """
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    # Candidate indices for replacement: nouns, verbs, adjectives, adverbs
    candidates = [i for i,(w,tag) in enumerate(pos_tags)
                  if tag.startswith(('N','V','J','R')) and w.isalpha()]
    random.shuffle(candidates)
    num_replaced = 0

    for idx in candidates:
        if num_replaced >= n_replacements:
            break
        if random.random() > replace_prob:
            continue
        word = words[idx]
        tag = pos_tags[idx][1]
        syns = get_synonyms(word, pos_tag=tag)
        if not syns:
            continue
        # randomly pick a synonym
        new_word = random.choice(syns)
        words[idx] = new_word
        num_replaced += 1

    return ' '.join(words)

def process_ptb(input_path, output_path, n_augments=1, n_replacements=1, replace_prob=0.1):
    """
    Read PTB sentences from a file, perform augmentation and write output.
    Assumes one sentence per line in input_path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            sentence = line.strip()
            if not sentence:
                continue
            # write original
            f_out.write(sentence + '\n')
            # generate augmentations
            for _ in range(n_augments):
                aug = synonym_replacement(sentence,
                                          n_replacements=n_replacements,
                                          replace_prob=replace_prob)
                f_out.write(aug + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augment PTB with synonym replacement (WordNet + OMW)")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to input PTB file (one sentence per line).")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to write augmented dataset (one sentence per line).")
    parser.add_argument('--n_augments', type=int, default=1,
                        help="Number of augmented versions per original sentence.")
    parser.add_argument('--n_replacements', type=int, default=1,
                        help="Max number of words to replace per sentence.")
    parser.add_argument('--replace_prob', type=float, default=0.1,
                        help="Probability of each eligible word being replaced.")
    args = parser.parse_args()

    process_ptb(input_path=args.input,
                output_path=args.output,
                n_augments=args.n_augments,
                n_replacements=args.n_replacements,
                replace_prob=args.replace_prob)
