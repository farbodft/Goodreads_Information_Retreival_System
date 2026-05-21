import json
# from indexes_enum import Indexes,Index_types
# from index_reader import Index_reader
from Logic.indexer.indexes_enum import Indexes, Index_types
from Logic.indexer.index_reader import Index_reader


class DocumentLengthsIndex:
    def __init__(self, path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """

        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.CHARACTERS: self.get_documents_length(Indexes.CHARACTERS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS: self.get_documents_length(Indexes.DESCRIPTIONS.value)
        }
        self.store_document_lengths_index(path, Indexes.CHARACTERS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.DESCRIPTIONS)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        lengths = {}

        for doc_id, doc in self.documents_index.items():
            field_data = doc.get(where)

            if isinstance(field_data, list):
                # for characters and genres, length is the number of elements of the lists
                lengths[doc_id] = len(field_data)
            elif isinstance(field_data, str):
                # for description, length is the number of tokens
                lengths[doc_id] = len(field_data.split())
            else:
                lengths[doc_id] = 0

        return lengths

    def store_document_lengths_index(self, path, index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)


if __name__ == '__main__':
    indexes_path = 'index/'
    document_lengths_index = DocumentLengthsIndex(indexes_path)
    print('Document lengths index stored successfully.')
