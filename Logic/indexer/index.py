import time
import os
import json
import copy
from Logic.indexer.indexes_enum import Indexes
# from indexes_enum import Indexes



class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.CHARACTERS.value: self.index_characters(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.DESCRIPTIONS.value: self.index_descriptions(),
        }

        # sort indexes
        self.sort_index()

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = str(doc.get('id'))
            current_index[doc_id] = doc

        return current_index

    def index_characters(self):
        """
        Index the documents based on the characters.

        Returns
        ----------
        dict
            The index of the documents based on the characters. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for doc in self.preprocessed_documents:
            doc_id = str(doc.get('id'))
            characters = doc.get('characters', [])
            for char in characters:
                if not char:
                    continue
                # seeing a character for the first time
                if char not in index:
                    index[char] = {}
                # increment tf
                index[char][doc_id] = index[char].get(doc_id, 0) + 1

        return index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for doc in self.preprocessed_documents:
            doc_id = str(doc.get('id'))
            genres = doc.get('genres', [])
            for genre in genres:
                if not genre:
                    continue
                # seeing a genre for the first time
                if genre not in index:
                    index[genre] = {}
                # increment tf
                index[genre][doc_id] = index[genre].get(doc_id, 0) + 1

        return index

    def index_descriptions(self):
        """
        Index the documents based on the descriptions.

        Returns
        ----------
        dict
            The index of the documents based on the descriptions. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = str(doc.get('id'))
            description = doc.get('description', "")
            # tokenize the description
            words = description.split()
            for word in words:
                if not word:
                    continue
                # seeing a word for the first time
                if word not in current_index:
                    current_index[word] = {}
                # increment tf
                current_index[word][doc_id] = current_index[word].get(doc_id, 0) + 1

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, characters, genres, descriptions)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            target_index = self.index.get(index_type, {})
            posting_dict = target_index.get(word, {})
            return list(posting_dict.keys())
        except:
            return []

    def sort_index(self):
        """
        function to sort the posting lists for each term based on doc_id for further use
        """
        indexes = [Indexes.CHARACTERS.value, Indexes.GENRES.value, Indexes.DESCRIPTIONS.value]
        for index_type in indexes:
            for term in self.index[index_type]:
                # sort items based on doc_id (as an integer)
                sorted_items = sorted(self.index[index_type][term].items(), key=lambda x: int(x[0]))
                self.index[index_type][term] = dict(sorted_items)

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        doc_id = str(document.get('id'))

        # add to document index
        self.index[Indexes.DOCUMENTS.value][doc_id] = document

        # add to characters index
        for char in document.get('characters', []):
            if char not in self.index[Indexes.CHARACTERS.value]:
                self.index[Indexes.CHARACTERS.value][char] = {}
            self.index[Indexes.CHARACTERS.value][char][doc_id] = self.index[Indexes.CHARACTERS.value][char].get(doc_id,
                                                                                                                0) + 1

        # add to genre index
        for genre in document.get('genres', []):
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            self.index[Indexes.GENRES.value][genre][doc_id] = self.index[Indexes.GENRES.value][genre].get(doc_id, 0) + 1

        # add to descriptions index
        description = document.get('description', "")
        for word in description.split():
            if not word:
                continue
            if word not in self.index[Indexes.DESCRIPTIONS.value]:
                self.index[Indexes.DESCRIPTIONS.value][word] = {}
            self.index[Indexes.DESCRIPTIONS.value][word][doc_id] = self.index[Indexes.DESCRIPTIONS.value][word].get(
                doc_id, 0) + 1

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        document = self.index[Indexes.DOCUMENTS.value].get(document_id)
        if not document:
            return

        def remove_from_inverted(index_key, terms):
            idx = self.index[index_key]
            for term in terms:
                if term in idx and document_id in idx[term]:
                    del idx[term][document_id]
                    if not idx[term]:  # clean up empty term keys
                        del idx[term]

        remove_from_inverted(Indexes.CHARACTERS.value, document.get('characters', []))
        remove_from_inverted(Indexes.GENRES.value, document.get('genres', []))
        remove_from_inverted(Indexes.DESCRIPTIONS.value, document.get('description', "").split())

        # remove from metadata index
        del self.index[Indexes.DOCUMENTS.value][document_id]

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if not index_before_add[index].__contains__(key):
            index_before_add[index].setdefault(key, {})

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'characters': ['sandman', 'robin'],
            'genres': ['mystery', 'crime'],
            'description': 'good'
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'sandman')

        if (set(index_after_add[Indexes.CHARACTERS.value]['sandman']).difference(
                set(index_before_add[Indexes.CHARACTERS.value]['sandman']))
                != {dummy_document['id']}):
            print('Add is incorrect, sandman')
            return

        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'robin')

        if (set(index_after_add[Indexes.CHARACTERS.value]['robin']).difference(
                set(index_before_add[Indexes.CHARACTERS.value]['robin']))
                != {dummy_document['id']}):
            print('Add is incorrect, robin')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'mystery')

        if (set(index_after_add[Indexes.GENRES.value]['mystery']).difference(
                set(index_before_add[Indexes.GENRES.value]['mystery']))
                != {dummy_document['id']}):
            print('Add is incorrect, mystery')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'crime')

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        self.check_if_key_exists(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        if (set(index_after_add[Indexes.DESCRIPTIONS.value]['good']).difference(
                set(index_before_add[Indexes.DESCRIPTIONS.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'sandman')
        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'robin')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'mystery')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str = 'index/',
                    index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, characters, genres, descriptions)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        file_path = os.path.join(path, f"{index_name}_index.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.index[index_name]))

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        indexes = [Indexes.DOCUMENTS, Indexes.CHARACTERS, Indexes.GENRES, Indexes.DESCRIPTIONS]
        for index_name in indexes:
            file_path = os.path.join(path, f"{index_name.value}_index.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.index[index_name.value] = json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        print('comparing indexes')

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the descriptions
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False


def main():
    # load after lsh deduplicated data
    input_file = '../deduplicated_preprocessed.json'
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # initialize Indexer
        indexer = Index(documents)

        # step 1: run correctness checks
        print("--- Testing Index Quality ---")
        indexer.check_if_indexing_is_good(Indexes.DESCRIPTIONS.value, 'book')

        print("\n--- Testing Dynamic Updates ---")
        indexer.check_add_remove_is_correct()

        # step 2: store indexes
        print("\n--- Storing Indexes ---")
        storage_path = 'index/'
        for idx_type in [Indexes.DOCUMENTS, Indexes.CHARACTERS, Indexes.GENRES, Indexes.DESCRIPTIONS]:
            indexer.store_index(path=storage_path, index_name=idx_type.value)

        # step 3: test loading
        print("\n--- Verifying Load logic ---")
        loader = Index([])  # Start empty
        loader.load_index(storage_path)
        for idx_type in [Indexes.DOCUMENTS.value, Indexes.CHARACTERS.value, Indexes.GENRES.value,
                         Indexes.DESCRIPTIONS.value]:
            if indexer.check_if_index_loaded_correctly(idx_type, loader.index[idx_type]):
                print(f"Success: {idx_type} index matches disk content.")

    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run LSH.py first.")


if __name__ == '__main__':
    main()
