import string
from collections import deque
from typing import Callable, List, Tuple, Dict


class Snippet:
    """
    A class to generate relevant text snippets from documents based on a search query.
    
    It uses a single-pass sliding window with a 'lag' mechanism to ensure keywords
    at the very beginning and very end of documents are treated as potential centers.
    """

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable,
                 number_of_words_on_each_side: int = 5):
        """
        Initialize the Snippet generator.

        Args:
            normalize_function (Callable): A function that takes a word and returns its stemmed/normalized version.
            remove_stopword_function (Callable): A function that takes a query string and returns a list of filtered tokens.
            number_of_words_on_each_side (int): Number of words to include to the left and right of a keyword.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = (2 * number_of_words_on_each_side) + 1

    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Main orchestrator for snippet generation.

        Parameters:
            raw_doc (str): The original document string.
            query (str): The user's search query string.

        Returns:
            final_snippet (str): The formatted snippet with '***' highlighting and '...' separators.
            not_exist_words (list): The list of words from the query that were not found in the document.
        """
        if not raw_doc or not query:
            return "", []

        # tokenize raw_doc by space
        doc_tokens = raw_doc.split()

        # remove punctuations and normalize
        normalized_cache = []
        for token in doc_tokens:
            cleaned = token.strip(string.punctuation)
            normalized_cache.append(self.normalize(cleaned))

        # process query
        query_words = self.remove_stopword(query)
        query_set = set()
        for word in query_words:
            cleaned_q = word.strip(string.punctuation)
            norm_q = self.normalize(cleaned_q)
            if norm_q:
                query_set.add(norm_q)

        # find missing query words in text
        not_exist_words = []
        for word in query_words:
            cleaned_q = word.strip(string.punctuation)
            norm_q = self.normalize(cleaned_q)
            if norm_q not in normalized_cache and word not in not_exist_words:
                not_exist_words.append(word)

        # extract density coordinates, merge collisions, and create snippet
        best_windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        merged_windows = self._merge_windows(best_windows)
        final_snippet = self._create_snippet_text(doc_tokens, normalized_cache, merged_windows, query_set)

        return final_snippet, not_exist_words

    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        """
        Uses a sliding window to score the 'density' of query matches.
        
        Parameters:
            doc_tokens (list): List of original words from the document.
            normalized_cache (list): List of the same words, but normalized/stemmed.
            query_set (set): Set of normalized query stems.

        Returns:
            list: A list of (start_index, end_index) for the best windows found.
        """
        l = len(doc_tokens)
        if l == 0:
            return []

        best_score = -1
        all_windows = []

        for i in range(l):
            start = max(0, i - self.number_of_words_on_each_side)
            end = min(l - 1, i + self.number_of_words_on_each_side)

            # calculate density score (number of query hits inside range)
            score = sum(1 for idx in range(start, end + 1)
                        if normalized_cache[idx] in query_set and normalized_cache[idx] != "")

            if score > best_score:
                best_score = score
            all_windows.append((start, end, score))
        # no query hits in the document
        if best_score <= 0:
            return [(0, min(l - 1, self.win_size - 1))]

        # get all windows that match the highest density score
        best_windows = [(w[0], w[1]) for w in all_windows if w[2] == best_score]
        return best_windows

    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Combines window ranges that overlap or touch.
        
        Parameters:
            windows (list): List of (start, end) index tuples.

        Returns:
            list: List of merged (start, end) index tuples.
        """
        if not windows:
            return []

        # sort based on start index
        sorted_windows = sorted(windows, key=lambda x: x[0])
        merged = [sorted_windows[0]]

        for current_start, current_end in sorted_windows[1:]:
            last_start, last_end = merged[-1]

            # current window overlaps or touches the previous window
            if current_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        return merged

    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list,
                             merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        """
        Constructs the final formatted snippet string.
        
        Parameters:
            doc_tokens (list): Original document tokens.
            normalized_cache (list): Stemmed document tokens.
            merged_windows (list): Merged (start, end) indices.
            query_set (set): Normalized query stems.

        Returns:
            str: The final snippet with highlights and ellipses. 
                example: "The ***wizard*** went to ***Hogwarts.*** The ***wizard*** loved magic."

        """
        if not merged_windows:
            return ""

        parts = []
        l = len(doc_tokens)

        for i, (start, end) in enumerate(merged_windows):
            if i == 0 and start > 0:
                parts.append("...")
            elif i > 0 and start > merged_windows[i - 1][1] + 1:
                parts.append("...")

            window_tokens = []
            for idx in range(start, end + 1):
                token = doc_tokens[idx]
                norm_token = normalized_cache[idx]

                # check for query matches and wrap them with ***
                if norm_token in query_set and norm_token != "":
                    window_tokens.append(f"***{token}***")
                else:
                    window_tokens.append(token)
            parts.append(" ".join(window_tokens))
        # append ellipsis with ... if the window clip ends before the document text
        if merged_windows[-1][1] < l - 1:
            parts.append("...")

        return " ".join(parts)
