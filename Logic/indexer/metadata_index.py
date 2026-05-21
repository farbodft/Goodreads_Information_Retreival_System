from Logic.indexer.index_reader import Index_reader
from Logic.indexer.indexes_enum import Indexes, Index_types
# from index_reader import Index_reader
# from indexes_enum import Indexes, Index_types
import json


class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.path = path
        self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self, path):
        """
        Reads the documents.
        
        """
        reader = Index_reader(path, index_name=Indexes.DOCUMENTS)
        self.documents = reader.index

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'characters': self.get_average_document_field_length('characters'),
            'genres': self.get_average_document_field_length('genres'),
            'descriptions': self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index

    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        if not self.documents:
            return 0

        total_length = 0
        doc_count = len(self.documents)

        for doc in self.documents.values():
            field_data = doc.get(where)
            if field_data:
                if isinstance(field_data, list):
                    # characters and genres
                    total_length += len(field_data)
                elif isinstance(field_data, str):
                    # descriptions
                    total_length += len(field_data.split())
        return total_length / doc_count if doc_count > 0 else 0

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


if __name__ == "__main__":
    meta_index = Metadata_index('index/')
