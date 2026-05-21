import pickle
import os
import string


class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        padded = f"${word}$"
        if len(padded) < k:
            return {padded}
        return {padded[i:i + k] for i in range(len(padded) - k + 1)}

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        if not first_set or not second_set:
            return 0.0

        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)

        return len(intersection) / len(union)

    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_k_gram_words = {}
        word_counter = {}

        for doc in all_documents:
            if not isinstance(doc, str):
                continue
            words = doc.split()
            for word in words:
                # remove punctuation and case fold
                cleaned_word = word.lower().strip(string.punctuation)
                if not cleaned_word:
                    continue

                # update corpus frequency count
                word_counter[cleaned_word] = word_counter.get(cleaned_word, 0) + 1

                # build k-gram sets for unique vocabulary terms
                if cleaned_word not in all_k_gram_words:
                    all_k_gram_words[cleaned_word] = self.k_gram_word(cleaned_word)

        return all_k_gram_words, word_counter

    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        cleaned_word = word.lower().strip(string.punctuation)
        if not cleaned_word:
            return []

        word_grams = self.k_gram_word(cleaned_word)
        candidates = []

        for vocab_word, vocab_grams in self.all_k_gram_words.items():
            score = self.jaccard_score(word_grams, vocab_grams)
            frequency = self.word_counter.get(vocab_word, 0)
            candidates.append((vocab_word, score, frequency))

        # sort by score descending by score, then frequency and then alphabetically
        candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))

        return [item[0] for item in candidates[:5]]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        if not query:
            return ""

        # remove white spaces from start and end of the string
        cleaned_query = query.strip()

        words = cleaned_query.split()
        corrected_words = []

        for word in words:
            cleaned_word = word.lower().strip(string.punctuation)
            if not cleaned_word:
                corrected_words.append(word)
                continue
            # ignore if the clean word is already valid
            if cleaned_word in self.word_counter:
                corrected_words.append(cleaned_word)
            else:
                # find the best replacement from candidates
                nearest = self.find_nearest_words(cleaned_word)
                if nearest:
                    corrected_words.append(nearest[0])
                else:
                    corrected_words.append(cleaned_word)

        return " ".join(corrected_words)
