import numpy as np
import math
from collections import defaultdict

class BM25new:
    def __init__(self, index, index_DL, k1=2, b=0.75):
        self.index = index
        self.index_DL = index_DL
        self.k1 = k1
        self.b = b
        self.N = len(index_DL)
        self.AVGDL = np.mean(list(index_DL.values()))

    def _calc_idf(self, term_frequencies):
        idf = {}
        for term, _ in term_frequencies.items():
            n_ti = self.index.df.get(term, 0)
            idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
        return idf

    def fetch_postings_lists(self, query):
        term_frequencies = defaultdict(dict)
        for term in np.unique(query):
            if term in self.index.df:
                postings = self.index.read_a_posting_list("", term, "nettaya-315443382")
                for doc_id, freq in postings:
                    term_frequencies[term][doc_id] = freq
        return term_frequencies

    def _score(self, query, doc_id, term_frequencies, idf):
        score = 0.0
        doc_len = self.index_DL.get(doc_id, 0)
        for term in query:
            freq = term_frequencies.get(term, {}).get(doc_id, 0)
            numerator = idf.get(term, 0) * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
            score += numerator / denominator if denominator != 0 else 0
        return score

    def search(self, query, N=100):
        term_frequencies = self.fetch_postings_lists(query)
        idf = self._calc_idf(term_frequencies)
        candidates = set(doc_id for postings in term_frequencies.values() for doc_id in postings)
        scores = [(doc_id, self._score(query, doc_id, term_frequencies, idf)) for doc_id in candidates]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:N]
