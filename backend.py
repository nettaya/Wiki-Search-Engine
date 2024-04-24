import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import nltk
from flask import logging
from nltk.stem.porter import PorterStemmer
import os
import re
from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import numpy as np
import math
from operator import add
import builtins
import csv
from collections import defaultdict
from inverted_index_gcp import *
import hashlib
import math
from itertools import chain
import time
from BM25 import BM25
from io import BytesIO
# At the start of backend.py
nltk.download('wordnet')
# Ensure that NLTK stopwords are downloaded
nltk.download('stopwords')
from nltk.corpus import wordnet as wn


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = stopwords_frozen.union(corpus_stopwords)
porter_stemmer = PorterStemmer()


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

class search_engine():
    def __init__(self):
        bucket_name = "nettaya-315443382"
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for blob in blobs:
            if (blob.name == 'postings_gcp_title/index_title.pkl'):
                with blob.open("rb") as f:
                    self.index_title = pickle.load(f)
            if (blob.name == 'postings_gcp_anchor/index_anchor.pkl'):
                with blob.open("rb") as f:
                    self.index_anchor_text = pickle.load(f)
            if (blob.name == 'index_body_DL.pkl'):
                with blob.open("rb") as f:
                    self.index_body_DL = pickle.load(f)
            if (blob.name == "id_title_index_id_title.pkl"):
                with blob.open("rb") as f:
                    self.id_title_dict = pickle.load(f)
            if (blob.name == "postings_gcp_body_posting_locs/index_body_posting_locs.pkl"):
                with blob.open("rb") as f:
                    self.index_body_posting_locs = pickle.load(f)

        bucket_name = "netta-315443382"
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for blob in blobs:
            if (blob.name == 'postings_gcp_body/index_body.pkl'):
                with blob.open("rb") as f:
                    self.index_body = pickle.load(f) #df+term_total

    def search(self, query, N=30):
        """
        Executes a search for a query after preprocessing.
        """
        query_process = self.preprocess_query(query, False)

        if len(query_process) <= 2:
             final_res = self.search_combined_short_queries(query_process, self.index_title, self.index_anchor_text)
            # bm25_title = BM25(self.index_title,self.index_title.DL)
            # bm25_anchor= BM25(self.index_anchor_text,self.index_anchor_text.DL)
            # title_results = bm25_title.search(query_process, N)
            # anchor_results = bm25_anchor.search(query_process, N)
            # final_res = self.merge_results_BM25_short(title_results, anchor_results)
            # final_res = self.combine_search_BM25_short(final_res_BM25,results_combined)

        else:
            query_process_body = self.preprocess_query(query, True)
            bm25_body = BM25(self.index_body,self.index_body_DL.DL)
            bm25_title = BM25(self.index_title, self.index_title.DL)
            body_results = bm25_body.search(query_process_body, N)
            title_results = bm25_title.search(query_process, N)
            final_res = self.merge_results_BM25_long(body_results, title_results)
        return final_res[:N]

    def combine_search_BM25_short(self, final_res_BM25, results_combined,N=30):
        all_results = defaultdict(float)
        for doc_id, score in final_res_BM25:
            all_results[doc_id] += score

        for doc_id, score in results_combined:
            if doc_id in all_results:
                all_results[doc_id] = max(all_results[doc_id], score)
            else:
                all_results[doc_id] = score

        sorted_all_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:N]

        # Convert doc_ids to strings and fetch titles
        final_results = [[str(doc_id), self.id_title_dict.DL.get(int(doc_id), "Title Not Found")] for doc_id, _ in
                         sorted_all_results]

        return final_results

    def calculate_idf(self, term, index_title, index_anchor, total_docs):
        # Retrieve document frequency for the term from both indices and sum them
        df_title = index_title.df.get(term, 0)
        df_anchor = index_anchor.df.get(term, 0)
        df_combined = df_title + df_anchor

        # Calculate IDF using the combined document frequency
        return math.log((total_docs - df_combined + 0.5) / (df_combined + 0.5) + 1) if df_combined else 0

    def search_combined_short_queries(self, query_process, index_title, index_anchor, N=30, bucket_name="nettaya-315443382"):
        total_docs = len(index_title.DL) + len(index_anchor.DL)

        scores = defaultdict(float)


        for term in query_process:
            idf = self.calculate_idf(term, index_title, index_anchor, total_docs)

            # Score based on title index postings
            for doc_id, freq in index_title.read_a_posting_list("", term, bucket_name):
                scores[str(doc_id)] += freq * idf * 0.7  # Assuming title weight is 0.7

            # Score based on anchor index postings
            for doc_id, freq in index_anchor.read_a_posting_list("", term, bucket_name):
                scores[str(doc_id)] += freq * idf * 0.3  # Assuming anchor weight is 0.3

        # Sorting documents by their scores in descending order and selecting top N.
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:N]
        sorted_scores_with_titles = [[doc_id, self.id_title_dict.DL[int(doc_id)]] for doc_id, _ in sorted_scores]

        return sorted_scores_with_titles
    def preprocess_query(self,query, use_stemming=True):
        """
        Tokenize a query into a list of tokens, filtering out stopwords, and optionally apply stemming.

        Parameters:
        -----------
        query : string
            The query string to preprocess.
        use_stemming : bool, optional (default=True)
            Indicates whether stemming should be applied to the tokens.

        Returns:
        --------
        list
            A list of preprocessed tokens from the query.
        """
        # Tokenization and stopword removal
        tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]

        # Optionally apply stemming
        if use_stemming:
            tokens = [porter_stemmer.stem(token) for token in tokens]

        return tokens


    def expand_query_with_weights(self,query):
        # Preprocess the original query

        # Assign a higher weight to original query terms
        weighted_terms = {term: 0.6 for term in query}

        # Expand query with synonyms, assigning lower weight
        for term in query:
            for syn in wn.synsets(term):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace("_", " ")
                    if lemma_name not in weighted_terms:
                        weighted_terms[lemma_name] = 0.4

        return weighted_terms



    def merge_results_BM25_short(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=100):
        combined_scores = defaultdict(float)

        # Update combined scores with weighted title and body scores
        for doc_id, score in title_scores:
            combined_scores[doc_id] += score * title_weight
        for doc_id, score in body_scores:
            combined_scores[doc_id] += score * text_weight

        # Sort and select top N results
        sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:N]

        return sorted_combined_scores

    def merge_results_BM25_long(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=100):
        combined_scores = defaultdict(float)

        # Update combined scores with weighted title and body scores
        for doc_id, score in title_scores:
            combined_scores[doc_id] += score * title_weight
        for doc_id, score in body_scores:
            combined_scores[doc_id] += score * text_weight

        # Sort and select top N results
        sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:N]

        # Map document IDs to titles for the final output
        final_results = [(doc_id, self.id_title_dict.DL.get(doc_id, "Title Not Found")) for doc_id, _ in
                         sorted_combined_scores]
        return final_results




