
import datetime


from nltk.stem.wordnet import WordNetLemmatizer


def generate_past_n_days(numdays):
    """Generate N days until now, e.g., [20151231, 20151230]."""
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    return [x.strftime("%Y%m%d") for x in date_list]

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