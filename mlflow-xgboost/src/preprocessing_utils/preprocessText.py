# import re
# import spacy
# from stop_words import get_stop_words

# nlp = spacy.load("de_core_news_sm")

def preprocessText(text):
    # doc = nlp(text)
    # tokensWithoutStopwords = [token.text for token in doc if not token.is_stop]
    # stemmedTokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # return " ".join(stemmedTokens)
    text = "this is a test to see if code_path works!?"
    return text