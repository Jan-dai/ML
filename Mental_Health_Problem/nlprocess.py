import json , os , spacy
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from tqdm import tqdm

def load_data2dataframe(file):
    """
    讀取 JSON 文件並將其轉換為 Pandas DataFrame。
    參數:
        file (str): 檔案路徑。

    錯誤回報:
        ValueError: 如果在資料集目錄中找不到 JSON 檔案。
        ValueError: 如果指定的 JSON 檔案不存在。
        
    回傳:
        pd.DataFrame: 包含文本、壓力源類別、壓力源詞和時間間隔的 DataFrame。
    """
    if not os.path.exists(file):
        raise ValueError(f"File '{file}' does not exist.")
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in JSON file, got {type(data)}")
        temp = []
        for item in data:
            #print(item)
            TEXT = item["text"]
            INTERVAL = item["interval"]
            try:
                STRESSOR_class = item["labels"][0][0]
                STRESSOR_WORD = item["labels"][0][4]
            except IndexError:
                STRESSOR_class = np.nan
                STRESSOR_WORD = np.nan
            temp.append((TEXT, STRESSOR_class, STRESSOR_WORD, INTERVAL))

    # Convert list of tuples to DataFrame with column names
    return pd.DataFrame(temp, columns=["text", "stressor_class", "stressor_word", "interval"])

def normalize_spacy(text , nlp = None):
    '''
    將文本轉換為小寫,並移除標點符號、空白和停用詞，在進行詞型還原。
    '''
    doc = nlp(text.lower())
    lemmatized = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop:
            continue
        lemma = token.lemma_
        lemmatized.append(lemma)
    return lemmatized

def to_corpus(df: pd.DataFrame , column_name: str , type: str , nlp = None):
    '''
    '''
    # 1. 文字正規化
    texts = [normalize_spacy(doc, nlp) for doc in tqdm(df[column_name], desc="Normalizing")]

    # 2. 過濾空文本（避免 dictionary 出錯）
    df["normalized"] = texts
    df = df[df["normalized"].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    # 3. 建立字典
    texts = df["normalized"].tolist()
    dictionary = corpora.Dictionary(texts)

    # 5. 過濾太少／太常見詞
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    # 6. 建立新語料（這時候 dictionary ID 是新的）
    if type == "bow":
        corpus = [dictionary.doc2bow(text) for text in tqdm(texts, desc="Building BOW Corpus")]
    elif type == "tfidf":
        bow_corpus = [dictionary.doc2bow(text) for text in tqdm(texts, desc="Building BOW Corpus")]
        tfidf = TfidfModel(bow_corpus)
        corpus = [tfidf[text_corpus] for text_corpus in tqdm(bow_corpus, desc="Building TFIDF Corpus")]

    # 7. 刪除corpus中為空的元素，並使df和corpus同步
    empty_indices = [i for i, doc in enumerate(corpus) if not doc]
    corpus = [doc for i, doc in enumerate(corpus) if i not in empty_indices]
    df = df.drop(empty_indices).reset_index(drop=True)
    
    return {"df": df, "texts": texts,"dictionary": dictionary, "corpus": corpus}