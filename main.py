import re
from nltk.corpus import stopwords as nltk_stopwords
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pymorphy3
import pandas as pd
import numpy as np
from pymystem3 import Mystem
from tqdm import tqdm
import time

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()


def lemmatize(doc, stopwords):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None

# Оставим в тексте только кириллические символы
def clear_text(text):
    clear_text = re.sub(r'[^А-яЁё]+', ' ', text).lower()
    return " ".join(clear_text.split())


# напишем функцию удаляющую стоп-слова
def clean_stop_words(text, stopwords):
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)


def lemmatize(df: (pd.Series, pd.DataFrame),
              text_column: (None, str),
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

    # загрузим список стоп-слов
    stopwords = set(nltk_stopwords.words('russian'))
    np.array(stopwords)

    labeled_tweets.columns = ['text', 'label']

    # test + clear
    start_clean = time.time()
    labeled_tweets['text_clear'] = labeled_tweets['text'].apply(lambda x: clean_stop_words(clear_text(str(x)), stopwords))
    print('Обработка текстов заняла: ' + str(round(time.time() - start_clean, 2)) + ' секунд')



    # lemmatize
    print(labeled_tweets)
    labeled_tweets['lemm_clean_text'] = lemmatize(
        df=labeled_tweets,
        text_column='text_clear',
        n_samples=100,
        break_str='br',
    )



    train, test = train_test_split(labeled_tweets,
                                   test_size=0.2,
                                   random_state=12348,
                                   )

    print(train.shape)
    print(test.shape)


    # selected['text_clear'] = selected['text'] \
    #     .apply(lambda x:
    #            clean_stop_words(
    #                clear_text(str(x)),
    #                stopwords))
    #
    # print('Обработка текстов заняла: ' + str(round(time.time() - start_clean, 2)) + ' секунд')
    #
    # selected['lemm_clean_tex'] = lemmatize(
    #     df=selected,
    #     text_column='text_clear',
    #     n_samples=100,
    #     break_str='br',
    # )

