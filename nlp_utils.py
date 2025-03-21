import re
import nltk
import spacy
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

def cleaning(user_text):
    user_text = user_text.lower()
    user_text = re.sub(r"[^a-zA-Z\s]", "", user_text)
    tokens = word_tokenize(user_text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def extract_words(user_text):
    if nlp is None:
        words = user_text.lower().split()
        return {w:1.0 for w in words if w not in stopwords.words("english")}
    try:
        doc = nlp(user_text)
        extracted = {ent.text.lower(): ent.label_ for ent in doc.ents}
        if not extracted:
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                    extracted[token.text.lower()] = token.pos_
        return extracted
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return {w:1.0 for w in user_text.lower().split() if w not in stopwords.words("english")}

def get_top_keywords(texts, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:num_keywords]
    return top_n.tolist()


