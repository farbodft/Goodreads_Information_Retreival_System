import re
import string
import json
import csv
from nltk.stem import PorterStemmer


class Preprocessor:
    def __init__(self, custom_stopwords_path='stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        # compile patterns
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_pattern = re.compile(pattern)

        # load stopwords
        with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
            self.stopwords = set([line.strip() for line in f])

        # generate translate table to convert punctuation to space
        self.punc_table = str.maketrans({ch: ' ' for ch in string.punctuation})

        # set stemmer
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        if not text:
            return ""

        # remove urls
        text = self.url_pattern.sub('', text)
        # case-folding
        text = text.lower()
        # handle punctuations
        text = text.translate(self.punc_table)
        # normalize (remove stop words and stemming)
        words = text.split()
        filtered = [self.normalize(w) for w in words if w not in self.stopwords]
        return " ".join(filtered)

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return filtered

    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        str
            The normalized word.
        """
        word = self.stemmer.stem(word)  # apply stemming
        return word

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        return [self.preprocess_text(doc) for doc in documents]


def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()

    # handle text fields and phrase fields separately
    text_fields = ['title', 'description', 'author']
    phrase_fields = ['genres', 'characters']

    for doc in docs:
        for field in text_fields:
            if field in doc and doc[field]:
                # handle list field type
                if isinstance(doc[field], list):
                    doc[field] = [
                        preprocessor.preprocess_text(str(item))
                        for item in doc[field]
                    ]
                # handle string field type
                elif isinstance(doc[field], str):
                    doc[field] = preprocessor.preprocess_text(str(doc[field]))
        for field in phrase_fields:
            if field in doc and doc[field]:
                if isinstance(doc[field], list):
                    doc[field] = [str(item).strip().lower() for item in doc[field] if str(item).strip()]
                elif isinstance(doc[field], str):
                    doc[field] = [str(doc[field]).strip().lower()]


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    data = []

    def split_by_comma(value):
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            doc = {
                "id": row.get("bookId"),
                "title": row.get("title"),
                "author": row.get("author"),
                "description": row.get("description"),
                "genres": split_by_comma(row.get("genres")),
                "characters": split_by_comma(row.get("characters")),
                "languages": split_by_comma(row.get("language")),
                "publish_date": row.get("publish_date"),
                "num_pages": int(float(row["num_pages"])) if row.get("num_pages") else None,
                "avg_rating": float(row["avg_rating"]) if row.get("avg_rating") else None
            }

            data.append(doc)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(data))


if __name__ == '__main__':
    csv_to_json('../top_3000_rated_books.csv', 'crawled.json')

    json_file_path = 'crawled.json'
    with open(json_file_path, "r", encoding="utf-8") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open('preprocessed.json', "w", encoding="utf-8") as file:
        file.write(json.dumps(docs))
