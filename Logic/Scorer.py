import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        doc_ids = set()
        for term in query:
            if term in self.index:
                doc_ids.update(self.index[term].keys())
        return list(doc_ids)

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term in self.idf:
            return self.idf[term]

        # get df to calculate idf
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0
        # calculate idf
        idf = math.log10(self.N / df)
        self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        return dict(Counter(query))

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        if not query:
            return {}
        doc_method, query_method = method.split('.')
        relevant_docs = self.get_list_of_documents(query)
        query_tfs = self.get_query_tfs(query)

        scores = {}
        for doc_id in relevant_docs:
            scores[doc_id] = self.get_vector_space_model_score(
                query, query_tfs, doc_id, doc_method, query_method
            )
        return scores

    def get_vector_space_model_score(
            self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        query_weights = {}
        for term in query_tfs:
            w_q = self._apply_tf(query_tfs[term], query_method[0])
            if query_method[1] == 't':
                w_q *= self.get_idf(term)
            query_weights[term] = w_q

        if query_method[2] == 'c':
            query_weights = self._cosine_normalize(query_weights)

        if not hasattr(self, '_normalized_docs_cache'):
            self._normalized_docs_cache = {}

        cache_key = (document_id, document_method, id(self.index))
        if cache_key in self._normalized_docs_cache:
            doc_weights = self._normalized_docs_cache[cache_key]
        else:
            doc_weights = {}
            for term, postings in self.index.items():
                if document_id in postings:
                    tf = postings[document_id]
                    w_d = self._apply_tf(tf, document_method[0])
                    if document_method[1] == 't':
                        w_d *= self.get_idf(term)
                    doc_weights[term] = w_d

            if document_method[2] == 'c':
                doc_weights = self._cosine_normalize(doc_weights)

            # cache to use later
            self._normalized_docs_cache[cache_key] = doc_weights

        score = 0
        for term in set(query):
            w_d = doc_weights.get(term, 0)
            w_q = query_weights.get(term, 0)
            # update score
            score += w_d * w_q

        return score

    def compute_scores_with_okapi_bm25(
            self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        if not query:
            return {}
        relevant_docs = self.get_list_of_documents(query)
        scores = {}
        for doc_id in relevant_docs:
            scores[doc_id] = self.get_okapi_bm25_score(
                query, doc_id, average_document_field_length, document_lengths
            )
        return scores

    def get_okapi_bm25_score(
            self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        # setting parameters
        k1 = 1.5
        b = 0.75
        score = 0

        dl = document_lengths.get(document_id, 0)
        avgdl = max(average_document_field_length, 0.0001)  # avoid div by zero

        for term in set(query):
            if term not in self.index:
                continue

            tf = self.index[term].get(document_id, 0)
            idf = self.get_idf(term)

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))

            # update score
            score += idf * (numerator / denominator)

        return score

    def compute_scores_with_unigram_model(
            self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        if not query:
            return {}
        self._prepare_collection_stats()
        relevant_docs = self.get_list_of_documents(query)
        scores = {}

        for doc_id in relevant_docs:
            scores[doc_id] = self.compute_score_with_unigram_model(
                query, doc_id, smoothing_method, document_lengths, alpha, lamda
            )
        return scores

    def compute_score_with_unigram_model(
            self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        score = 0
        doc_len = document_lengths.get(document_id, 0)
        vocab_size = len(self.index)

        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)

            if smoothing_method == 'laplace':
                prob = (tf + 1) / (doc_len + vocab_size)
            elif smoothing_method == 'jm':
                cf = self._collection_frequencies.get(term, 0)
                p_mle_doc = tf / doc_len if doc_len > 0 else 0
                p_mle_coll = cf / self._collection_length if self._collection_length > 0 else 0
                prob = (lamda * p_mle_doc) + ((1 - lamda) * p_mle_coll)
            else:
                prob = tf / doc_len if doc_len > 0 else 1e-9

            # use len to prevent underflow
            score += math.log(prob if prob > 0 else 1e-12)

        return score

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if mode == 'n':
            return tf
        elif mode == 'l':
            return 1 + math.log10(tf) if tf > 0 else 0
        return tf

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        norm = math.sqrt(sum(w ** 2 for w in weights.values()))
        if norm == 0:
            return weights
        return {k: v / norm for k, v in weights.items()}

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        if self._collection_frequencies is not None:
            return

        self._collection_frequencies = {}
        total_len = 0
        for term, postings in self.index.items():
            cf = sum(postings.values())
            self._collection_frequencies[term] = cf
            total_len += cf
        self._collection_length = total_len
