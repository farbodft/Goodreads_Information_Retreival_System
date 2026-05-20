import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

        self.shingle_to_index = None  # mapping for shingle to index
        self.characteristic_matrix = None  # variable to store characteristic matrix after building it once

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        tokens = document.split()
        for i in range(len(tokens) - k + 1):
            shingles.add(" ".join(tokens[i:i + k]))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # collect shingles and index them
        shingled_docs = [self.shingle_document(doc, 2) for doc in self.documents]
        all_shingles = sorted(list(set().union(*shingled_docs)))
        self.shingle_to_index = {sh: i for i, sh in enumerate(all_shingles)}

        # generate matrix
        num_shingles = len(all_shingles)
        num_docs = len(self.documents)
        matrix = np.zeros((num_shingles, num_docs), dtype=np.int8)

        # set values in matrix
        for d, sh_set in enumerate(shingled_docs):
            for sh in sh_set:
                matrix[self.shingle_to_index[sh], d] = 1

        self.characteristic_matrix = matrix  # save the result matrix for further use
        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        if self.characteristic_matrix is None:
            self.build_characteristic_matrix()  # build characteristic matrix for the first time

        # initialize signature matrix with infinity
        num_shingles, num_docs = self.characteristic_matrix.shape
        signature_matrix = np.full((self.num_hashes, num_docs), np.inf)

        # generate random hash function's coefficients using a large prime
        prime = 2 ** 31 - 1
        hash_coeffs = [(random.randint(1, prime - 1), random.randint(0, prime - 1))
                       for _ in range(self.num_hashes)]

        # iterate through shingles and construct signature matrix
        for r in range(num_shingles):
            # update signature for docs containing current shingle
            docs_with_shingle = np.where(self.characteristic_matrix[r, :] == 1)[0]
            if len(docs_with_shingle) > 0:
                # calculate row hash
                row_hashes = np.array([((a * r + b) % prime) for a, b in hash_coeffs])
                signature_matrix[:, docs_with_shingle] = np.minimum(
                    signature_matrix[:, docs_with_shingle],
                    row_hashes[:, np.newaxis]
                )

        return signature_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_hashes, num_docs = signature.shape
        if bands * rows_per_band > num_hashes:
            raise ValueError("bands * rows_per_bound can't exceed total number of hashes.")

        buckets = {}

        for b in range(bands):
            start_row = b * rows_per_band
            end_row = start_row + rows_per_band

            for d in range(num_docs):
                band_signature = tuple(signature[start_row:end_row, d])
                bucket_id = hash((b, band_signature))  # unique id per band/signature pair

                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(d)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes // num_bands)
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        if not first_set or not second_set:
            return 0.0
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

    def find_duplicates(self, threshold=0.85):
        """
        Added function to find duplicates based on the results of lsh and mini hashing
        """
        # running lsh to find candidate pairs
        buckets = self.perform_lsh()

        # shingle docs for jaccard verification
        duplicates_to_remove = set()
        shingled_docs = [self.shingle_document(doc, 2) for doc in self.documents]

        for doc_indices in buckets.values():
            if len(doc_indices) > 1:
                unique_docs = list(set(doc_indices))
                for i, j in itertools.combinations(unique_docs, 2):
                    # skip if doc is already in duplicates
                    if i in duplicates_to_remove or j in duplicates_to_remove:
                        continue

                    # verify with jaccard score
                    score = self.jaccard_score(shingled_docs[i], shingled_docs[j])
                    if score >= threshold:
                        duplicates_to_remove.add(max(i, j))  # remove the document with higher index

        return duplicates_to_remove


def main():
    # load fake data for testing
    try:
        print("Testing on fake data.")
        with open('LSHFakeData.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_docs = [" ".join(doc['descriptions']) for doc in test_data if 'descriptions' in doc]
        # initialize with 100 hashes (25 bands of 4 rows)
        lsh = MinHashLSH(test_docs, num_hashes=100)
        buckets = lsh.perform_lsh()
        lsh.jaccard_similarity_test(buckets, test_docs)
    except FileNotFoundError:
        print("LSHFakeData.json not found. Skipping test.")

    # remove duplicates from the preprocessed.json file and store the result in deduplicated_preprocessed.json
    try:
        print("Applying on real data.")
        with open('preprocessed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [str(doc.get('description', '')) for doc in data]

        lsh = MinHashLSH(texts, num_hashes=100)
        duplicates = lsh.find_duplicates(threshold=0.85)

        cleaned_data = [doc for i, doc in enumerate(data) if i not in duplicates]

        with open('deduplicated_preprocessed.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(cleaned_data))

        print(f"Results: {len(data)} books -> {len(cleaned_data)} books. ({len(duplicates)} removed)")

    except FileNotFoundError:
        print("preprocessed.json not found. Run preprocess.py first.")


if __name__ == '__main__':
    main()
