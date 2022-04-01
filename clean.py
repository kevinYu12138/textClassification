import re
import spacy
from nltk.stem import PorterStemmer,LancasterStemmer


nlp =spacy.load("en_core_web_sm")  # 加载预训练的模型

stemmer = PorterStemmer()

d1 = "We're having a.b.c.d or i.e. u.s.a and 16.2 and U.K. and others."
d2 = "We buy 12 apples, good apples, from Ms. ABC at 16.2 dollars/pound 24/7 from Monday-Friday. How's that?"
d3 = "1. I won't eat 1/12 of the #1 top cakes. I got 1.2 dollars or 1.2usd and a long-term/short-term goal"
d4 = "I lost 22 pounds at 9:30, 11:05 or 1:30pm or 2pm or 3pm or 1-2pm on 2016-01-02 or 01-02-2016."
d5 = "  --He's a dedicated person. \t He dedicated his time to work! \n --are you sure?"
d6 = "I am interested in these interesting things that interested me."
docs = [d1, d2, d3, d4, d5, d6]

# 定义文本预处理/清洗函数
def lemmatize(text):
    tuples = [(str(token), token.lemma_ ) for token in nlp(text)]
    # print(tuples)
    text = ' '.join([lemma if lemma != '-PRON-' else token for token, lemma in tuples])
    text = re.sub(' ([_-]) ', '\\1', text)  # 'short - term' -> 'short-term'
    return text

# print(lemmatize(d5))
# print(lemmatize(d6))

# 去除停用词
def text_cleaner(text, option='none'):
    stopwords = set(['i', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are'])
    text = text.lower()                                                # 1.lower strings
    text = text.replace("'s", 's')                                     # 2.handle contractions
    text = re.sub('([a-zA-Z])\\.([a-zA-Z])\\.*', '\\1\\2', text)       # 3. handle abbreviations
    text = re.sub('\\d+', '', text)                                    # 4. handle numbers
    text = re.sub('[^a-zA-Z0-9\\s_-]', ' ', text)                      # 5. remove punctuations
    if 'lemma' in option:
        text = lemmatize(text)                                         # 6. perform lemmatization
    elif 'stem' in option:
        text = ' '.join(stemmer.stem(x) for x in text.split(' '))      # 6. perform stemming
    text = ' '.join(x for x in text.split(' ') if x not in stopwords)  # 7. remove stopwords(stopwords类型是set而不是list，因为set的查找速度更快)
    text = ' '.join(x.strip('_-') for x in text.split())            # 8. remove spaces,'_' and '-'
    return text

# 文本清洗
# option = 'none'
option = 'lemmatization'
# option = 'stemming'

clean_docs = [text_cleaner(doc, option) for doc in docs]
for raw, clean in zip(docs, clean_docs):
    print('original doc:', raw)
    print('cleaned doc:', clean)
    print()


# 自定义sklearn的transformer class
# from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, RegressorMixin
from tqdm import tqdm

class TextCleaner(object):
    d1 = "We're having a.b.c.d or i.e. u.s.a and 16.2 and U.K. and others."
    d2 = "We buy 12 apples, good apples, from Ms. ABC at 16.2 dollars/pound 24/7 from Monday-Friday. How's that?"
    d3 = "1. I won't eat 1/12 of the #1 top cakes. I got 1.2 dollars or 1.2usd and a long-term/short-term goal"
    d4 = "I lost 22 pounds at 9:30, 11:05 or 1:30pm or 2pm or 3pm or 1-2pm on 2016-01-02 or 01-02-2016."
    d5 = "  --He's a dedicated person. \t He dedicated his time to work! \n --are you sure?"
    d6 = "I am interested in these interesting things that interested me."
    sample_docs = [d1, d2, d3, d4, d5, d6]

    def __init__(self, option='none'):
        self.option = option
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = PorterStemmer()

    def lemmatize(self, text):
        tuples = [(str(token), token.lemma_) for token in self.nlp(text)]
        text = ' '.join([lemma if lemma != '-PRON-' else token for token, lemma in tuples])
        text = re.sub(' ([_-]) ', '\\1', text)  # 'short - term' -> 'short-term'
        return text

    def text_cleanner(self, text):
        stopwords = set(['i', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are'])
        text = text.lower()  # 1.lower strings
        text = text.replace("'s", 's')  # 2.handle contractions
        text = re.sub('([a-zA-Z])\\.([a-zA-Z])\\.*', '\\1\\2', text)  # 3. handle abbreviations
        text = re.sub('\\d+', '', text)  # 4. handle numbers
        text = re.sub('[^a-zA-Z0-9\\s_-]', ' ', text)  # 5. remove punctuations
        if 'lemma' in self.option:
            text = self.lemmatize(text)  # 6. perform lemmatization
        elif 'stem' in self.option:
            text = ' '.join(self.stemmer.stem(x) for x in text.split(' '))  # 6. perform stemming
        text = ' '.join(x for x in text.split(' ') if
                        x not in stopwords)  # 7. remove stopwords(stopwords类型是set而不是list，因为set的查找速度更快)
        text = ' '.join(x.strip('_-') for x in text.split())  # 8. remove spaces,'_' and '-'
        return text

    def transform(self, docs):  # transformer必须要有fit()和transform(）方法
        clean_docs = []
        self.fails = []
        for doc in tqdm(docs):
            try:
                clean_docs.append(self.text_cleanner(doc))
            except:
                self.fails.append(doc)
        if len(self.fails) > 0:
            print("Some documents failed to be converted. Check self.fails for failed documents")
        return clean_docs

    def fit(self, docs, y=None):
        return self

    def fit_transform(self, docs, y=None):
        return self.fit(docs,y).transform(docs)

cleanner = TextCleaner(option='lemma')
docs = cleanner.sample_docs
cleaned_docs = cleanner.transform(docs)
for raw,clean in zip(docs, cleaned_docs):
    # print('original doc:', raw)
    print( clean)
    print()