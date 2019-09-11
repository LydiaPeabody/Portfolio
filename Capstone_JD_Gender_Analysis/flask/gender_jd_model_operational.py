import flask
import dill
import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#creates application that runs on a webserver
app = flask.Flask(__name__)

with open('../model_tfidf', 'rb') as f:
    PREDICTOR = dill.load(f)

@app.route("/")
def hello():
    return '''
    <body>
    <h2> Let's test a job description! </h2>
    </body>
    '''


@app.route('/model', methods=['POST', 'GET'])
def model():
    def text_stemmer(text):

        """
        Uses PorterStemmer on a text. Returns the stemmed text as a single string.
        """

        porter = PorterStemmer()
        stemmed_words = [porter.stem(word) for word in word_tokenize(text)]

        return(' '.join(stemmed_words))

    def spacy_lemmatizer(text):
        '''
        Uses Spacy's built-in lemmatizer on a text; returns the lemmatized text as a single string
        '''
        lemmatized_words = [word.lemma_ for word in nlp(text)]
        return(' '.join(lemmatized_words))


    def spacy_stopword_removal(text):
        '''
        Removes words on Spacy's stopwords list from text
        '''
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
        return(' '.join(tokens))

    def prepare_text(text):
        '''
        removes stopwords, lemmatizes, and stems text
        '''
        no_stops = spacy_stopword_removal(text)
        stemmed_lemmatized_text = text_stemmer(spacy_lemmatizer(no_stops))
        return stemmed_lemmatized_text

    if flask.request.method == 'POST':
        inputs = flask.request.form

        try:
            jd = inputs['job_description'][0:]
            prepped_jd = prepare_text(jd)
            item = pd.DataFrame([[prepped_jd]],
                                columns=['[prepped_jd]'])
            score = PREDICTOR.predict_proba([prepped_jd])
            female = int(score[0][0]*100)
            male = int(score[0][1]*100)
            # female = 0
            # male = 0
        except:
            female = 0
            male = 0

    else:
        female = 0
        male = 0

    return flask.render_template('dataentrypage.html', male = male, female = female)




if __name__ == '__main__':
    #debug = True helps autodetect changes in code without reloading
    nlp = spacy.load('en')
    app.run(debug=True)
