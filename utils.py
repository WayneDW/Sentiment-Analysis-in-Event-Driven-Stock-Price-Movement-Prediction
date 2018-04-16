
import time
import datetime
import numpy as np
from urllib.request import urlopen

from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup


def generate_past_n_days(numdays):
    """Generate N days until now, e.g., [20151231, 20151230]."""
    base = datetime.datetime.today()
    date_range = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    return [x.strftime("%Y%m%d") for x in date_range]

wordnet = WordNetLemmatizer()
def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    try:
        word = wordnet.lemmatize(word, 'v') # unify tense
    except:
        pass
    try:
        word = wordnet.lemmatize(word) # unify noun
    except:
        pass
    return word

def get_soup_with_repeat(url, repeat_times=3, verbose=True):
    for i in range(repeat_times): # repeat in case of http failure
        try:
            time.sleep(np.random.poisson(3))
            response = urlopen(url)
            data = response.read().decode('utf-8')
            return BeautifulSoup(data, "lxml")
        except Exception as e:
            if i == 0:
                print(e)
            if verbose:
                print('retry...')
            continue
