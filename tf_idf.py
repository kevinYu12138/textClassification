from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

d1 = 're have abcd or ie usa and and uk and other'
d2 = 'buy apple good apple from ms abc at dollar pound from monday-friday how s that'
d3 = 'win t eat of the top cake get dollar or usd and a long-term short-term goal'
d4 = 'lose pound at or pm or pm or pm or pm on  or '
d5 = 'hes a dedicated person dedicate his time to work are sure'
d6 = 'be interested in these interesting thing that interest'
docs = [d1, d2, d3, d4, d5, d6]
vec = TfidfVectorizer(
    input='content',                     # default；{‘filename’，‘file’，‘content’}；input must be a list
    encoding='utf-8',                    # default; same options as in str.decode(encoding='utf-8')
    decode_error='strict',               # default; same options as in str.decode(errors='strict')
    # preprocessing arguments
    lowercase=True,                      # default; convert strings to lowercase
    strip_accents='unicode',             # default=None; remove accents; 'unicode' is slower but universal
    preprocessor=None,                   # default
    # tokenization arguments
    analyzer='word',                     # default;{'word','char','char_wb'}
    token_pattern=u'(?u)\\b\\w\\w+\\b',  # default; equivalent to using a nltk.RegexpTokenizer()
    tokenizer=None,                      # default; can use a nltk tokenizer
    # vocabulary arguments
    stop_words=None,                     # default=None; try stop_words='english' for example
    ngram_range=(1,2),                   # default=(1,1)
    min_df=1,                            # default=1  ; int or [0.0, 1.0]; ignore terms with a doc-freq < cutoff
    max_df=0.8,                          # default=1.0; [0.0, 1.0] or int; ignore terms with a doc-freq > cutoff; 整数是多少个，小数是百分比
    max_features=None,                   # default; keep only the top max_features ordered by term-freq; 保持满足上述条件的词汇量大小
    vocabulary=None,                     # default; if provided, max_df, min_df ,max_features are ignored
    # TF - IDF adjustment arguments
    binary=False,                        # default; if True, all non-zero term counts are set to 1
    sublinear_tf=True,                   # default; if True, use 1 + log(tf) for non-zeros; else use tf
    use_idf=True,                        # default; if True, enable IDF re-weighting
    smooth_idf=True,                     # default; if True, use 1 + log((N_docs+1)/(df+1)), else use 1 + log(N_docs/df)
    norm='l2'                            # default; if True, preform post TF-IDF normalization such that output row has unit norm
)

# understand TfidfVectorizer - before performing vec.fit_transform()
preprocessor = vec.build_preprocessor()       #lower strings & remove accents
print(preprocessor("Let's TRY this OUT: aigué béonté"))

tokenizer = vec.build_tokenizer()        #tokenize documents
print(tokenizer("let's try this out: aigue beonte"))

analyzer = vec.build_analyzer()      # preprocess > tokenize > remove > stopwords > get n-grams > prune vocabulary
print(analyzer("let's try this out: aigue beonte"))

# understand TfidfVectorizer - after performing vec.fit_transform()
document_term_matrix = vec.fit_transform(docs)  # scipy sparse csr matrix
idx2term = vec.get_feature_names()
term2idx = vec.vocabulary_
# print(idx2term)
# print(term2idx)
vec.get_stop_words()
vec.stop_words_

document_term_matrix = pd.DataFrame(document_term_matrix.todense())  # convert DTMatrix to DataFrame
document_term_matrix.columns = vec.get_feature_names()
print(document_term_matrix)
