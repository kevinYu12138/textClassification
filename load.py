import spacy
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from nltk.stem import PorterStemmer,LancasterStemmer
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers', 'footers', 'quotes')
data = fetch_20newsgroups(subset='all', categories=categories, remove=remove, shuffle=True,random_state=2021)
label2name = {i: x for i, x in enumerate(data.target_names)}
name2label = {x: i for i, x in enumerate(data.target_names)}
df = pd.DataFrame({'label_name':data.target, 'label':data.target, 'text':data.data})
# print(df[:5])
df['label_name'] = df['label_name'].map(label2name)
# print(df[:5])
print(df.shape)

# 文本清洗
option = 'none'  # Speed: 'none'>'stem'>'lemma'
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

cleaner = TextCleaner(option=option)
# print(df['text'])
x = cleaner.transform(df['text'])
y = df['label'].astype('int16').to_list()
print(len(x), len(y))
# print(x[:5])

# 划分训练集和数据集
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.9, random_state=2021)
print('X_train shape:', len(X_train), '\t', 'Y_train shape:', len(Y_train))
print('X_test shape:', len(X_test), '\t', 'Y_test shape:', len(Y_test))

# tf-idf 创建 Document-Term Matrix
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.95,
    max_features=4000,
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
print('X_train shape:', X_train.shape, '\t', 'Y_train shape', len(Y_train))
print('X_test shape:', X_test.shape, '\t', 'Y_test shape', len(Y_test))