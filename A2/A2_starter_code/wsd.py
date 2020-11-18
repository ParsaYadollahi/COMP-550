# NLTK
from re import match
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Builtins
import re
import string


DATAENCODING = "utf-8"
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


class wsd:
    def __init__(self, wsd_instance) -> None:
        self.wsd_instance = wsd_instance
        self.lemma, self.preprocessed_context = self.basic_preprocess(
            wsd_instance)
        return

    def basic_preprocess(self, wsd_instance):

        preprocessed_context = ' '.join(
            [c.decode(DATAENCODING) for c in wsd_instance.context])

        text = preprocessed_context.lower()
        text = re.sub("\\W", " ", text)
        text = word_tokenize(text)

        # Remove stopwords and punctuation
        basic_preprocess_text = [
            word for word in text if word not in STOPWORDS and word not in string.punctuation
        ]

        lemmatized_text = []

        # tag with pos to get accurate results from lemmatizer
        for word, tag in pos_tag(basic_preprocess_text):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None

            if wntag:
                lemmatized_text.append(
                    LEMMATIZER.lemmatize(word, wntag))
            else:
                lemmatized_text.append(word)

        preprocessed_context = " ".join(lemmatized_text)

        lemma = wsd_instance.lemma.decode(DATAENCODING)
        lemma = lemma.lower()

        print(preprocessed_context)

        return lemma, preprocessed_context

    def most_frequent_sense_baseline(self):
        pred = set()
        lemma = self.lemma

        syns = wordnet.synsets(lemma)[0]

        for matchingSense in syns.lemmas():
            pred.add(matchingSense.key())

        return pred

    def run_lesk(self):
        pred = set()
        lemma = self.lemma
        context = self.preprocessed_context

        syns = lesk(context, lemma)

        # Add in all the possible keys
        for matchingSense in syns.lemmas():
            pred.add(matchingSense.key())

        return pred
