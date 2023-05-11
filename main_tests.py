import re
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# загрузим список стоп-слов
stopwords = set(nltk_stopwords.words('russian'))
np.array(stopwords)


# Оставим в тексте только кириллические символы
def clear_text(text):
    clear_text = re.sub(r'[^А-яЁё]+', ' ', text).lower()
    return " ".join(clear_text.split())


# напишем функцию удаляющую стоп-слова
def clean_stop_words(text, stopwords):
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)


def lemmatize(df,
              text_column,
              n_samples: int,
              break_str='br',
              ) -> pd.Series:
    """
    Принимает:
    df -- таблицу или столбец pandas содержащий тексты,
    text_column -- название столбца указываем если передаем таблицу,
    n_samples -- количество текстов для объединения,
    break_str -- символ разделения, нужен для ускорения,
    количество текстов записанное в n_samples объединяется
    в одит большой текст с предварительной вставкой символа
    записанного в break_str между фрагментами
    затем большой текст лемматизируется, после чего разбивается на
    фрагменты по символу break_str


    Возвращает:
    Столбец pd.Series с лемматизированными текстами
    в которых все слова приведены к изначальной форме:
    * для существительных — именительный падеж, единственное число;
    * для прилагательных — именительный падеж, единственное число,
    мужской род;
    * для глаголов, причастий, деепричастий — глагол в инфинитиве
    (неопределённой форме) несовершенного вида.

    """

    result = []

    m = Mystem()

    for i in tqdm(range((df.shape[0] // n_samples) + 1)):
        start = i * n_samples
        stop = start + n_samples

        sample = break_str.join(df[text_column][start: stop].values)

        lemmas = m.lemmatize(sample)
        lemm_sample = ''.join(lemmas).split(break_str)

        result += lemm_sample

    return pd.Series(result, index=df.index)


if __name__ == '__main__':
    positive = pd.read_csv('positive.csv',
                           sep=';',
                           header=None
                           )

    negative = pd.read_csv('negative.csv',
                           sep=';',
                           header=None
                           )

    positive_text = pd.DataFrame(positive.iloc[:, 3])
    negative_text = pd.DataFrame(negative.iloc[:, 3])

    positive_text['label'] = [1] * positive_text.shape[0]
    negative_text['label'] = [0] * negative_text.shape[0]

    labeled_tweets = pd.concat([positive_text, negative_text])

    labeled_tweets.index = range(labeled_tweets.shape[0])

    labeled_tweets.columns = ['text', 'label']

    # test + clear
    start_clean = time.time()
    labeled_tweets['text_clear'] = labeled_tweets['text'].apply(
        lambda x: clean_stop_words(clear_text(str(x)), stopwords))
    print('Обработка текстов заняла: ' + str(round(time.time() - start_clean, 2)) + ' секунд')

    # lemmatize

    labeled_tweets['lemm_clean_text'] = lemmatize(
        df=labeled_tweets,
        text_column='text_clear',
        n_samples=1000,
        break_str='br',
    )

    train, test = train_test_split(labeled_tweets,
                                   test_size=0.2,
                                   random_state=12348,
                                   )

    print(train.shape)
    print(test.shape)

    # Сравним распределение целевого признака
    for sample in [train, test]:
        print(sample[sample['label'] == 1].shape[0] / sample.shape[0])

    count_idf_positive = TfidfVectorizer(ngram_range=(1, 1))
    count_idf_negative = TfidfVectorizer(ngram_range=(1, 1))

    tf_idf_positive = count_idf_positive.fit_transform(train.query('label == 1')['text'])
    tf_idf_negative = count_idf_negative.fit_transform(train.query('label == 0')['text'])

    # Сохраним списки Idf для каждого класса
    positive_importance = pd.DataFrame(
        {'word': count_idf_positive.get_feature_names_out(),
         'idf': count_idf_positive.idf_
         }).sort_values(by='idf', ascending=False)

    negative_importance = pd.DataFrame(
        {'word': count_idf_negative.get_feature_names_out(),
         'idf': count_idf_negative.idf_
         }).sort_values(by='idf', ascending=False)

    print(positive_importance.query('word not in @negative_importance.word and idf < 10.8'))
    print(negative_importance.query('word not in @positive_importance.word and idf < 10'))

    fig = plt.figure(figsize=(12, 5))
    positive_importance.idf.hist(bins=100,
                                 label='positive',
                                 alpha=0.5,
                                 color='b',
                                 )
    negative_importance.idf.hist(bins=100,
                                 label='negative',
                                 alpha=0.5,
                                 color='r',
                                 )
