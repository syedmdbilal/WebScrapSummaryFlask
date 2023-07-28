
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

app = Flask(__name__)

def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        paragraph_texts = [p.get_text() for p in paragraphs]
        return ' '.join(paragraph_texts)
    else:
        return None

def extract_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    named_entities = [entity.text for entity in doc.ents]
    return named_entities

def summarize_text(text):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * 0.3)

    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return summary

def extract_noun_phrases(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        text = get_text_from_url(url)
        if text:
            named_entities = extract_named_entities(text)
            summary = summarize_text(text)
            noun_phrases = extract_noun_phrases(text)
            return render_template('result.html', text=text, named_entities=named_entities, summary=summary, noun_phrases=noun_phrases)
        else:
            return render_template('error.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
