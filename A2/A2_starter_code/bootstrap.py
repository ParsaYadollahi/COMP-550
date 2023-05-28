from nltk import word_tokenize as wt
from nltk.probability import *
from nltk.corpus import wordnet as wn
from functools import cmp_to_key
from collocation import Collocation


class Yarowsky:
    def __init__(self, lemma, context):
        self.lemma = lemma
        self.contexts = context
        self.seed = self.seed()
        self.decision_list = self.create_dlist(context)

    def create_dlist(self, contexts):
        decision_lists = {}
        collocations = {}

        for context in contexts:
            senses = []

            for word in wt(context):
                if word in self.seed.keys():
                    senses.append(self.seed[word])

            if len(senses) != 0:
                self.find_collocations(
                    context, senses, collocations
                )
                self.updated_seed(decision_lists, collocations)

        return decision_lists

    def seed(self):
        seed = {}
        for syns in wn.synsets(self.lemma):
            definition = syns.definition()
            seed[self.lemma] = syns

            for w in wt(definition):
                seed[w] = syns
        return seed

    def sort_collocations(self, collocations):
        return sorted(
            collocations.values(),
            key=cmp_to_key(
                lambda x, y: (
                    x.col() > y.col()) - (y.col() < x.col())
            ),
        )

    def updated_seed(self, decision_list, collocations):
        sorted = self.sort_collocations(collocations)

        for col in sorted:
            if col.col() > 0.01:
                decision_list[col.word] = col.sense()
                self.seed[col.word] = col.sense()

    def find_collocations(self, context, senses, collocations):
        for w in wt(context):
            if w != self.lemma:
                collocation = None

                if w in collocations:
                    collocation = collocations[w]
                else:
                    collocation = Collocation(w)

                for s in senses:
                    collocation.frequency[s] += 1

                collocations[w] = collocation

    def predict(self, context):
        sense_count = {}
        ret_set = set()

        best_synset = None
        best_count = 0

        for collocation in wt(context):
            if collocation not in self.decision_list:
                continue

            s_word = self.decision_list[collocation]

            if s_word not in sense_count:
                sense_count[s_word] = 0

            sense_count[s_word] += 1

            if sense_count[s_word] > best_count:
                best_count = sense_count[s_word]
                best_synset = s_word

        if best_synset is not None:
            return set([sense.key() for sense in best_synset.lemmas()])

        return ret_set
