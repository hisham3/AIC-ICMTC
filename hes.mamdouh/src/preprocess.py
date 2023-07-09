import tnkeeh as tn
from datasets import Dataset
import pandas as pd
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import qalsadi.lemmatizer
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import nltk
import json
import requests
import torch

def pre_text_cleaning(data, col="paragraph"):

    cleander = tn.Tnkeeh(
      normalize=True,
      remove_diacritics=True,
      # remove_special_chars=True,
      remove_english=True,
      remove_tatweel=True
    )

    return cleander.clean_hf_dataset(Dataset.from_dict(data), col).to_dict()


def segment_tokenize(data, col="paragraph", by="sent"):

    assert by != "word" or by != "sent", "You should choose to segment with word only or with sent"

    nltk.download("punkt")

    if by != "sent":
      data[col] = [word_tokenize(word) for word in data[col]]
    else:
      data[col] = [sent_tokenize(sent) for sent in data[col]]
      data[col] = [[word_tokenize(word) for word in sent] for sent in data[col]]

    return data

def remove_stopwords(data, col="paragraph", by="sent"):

    assert by != "word" or by != "sent", "You should choose to segment with word only or with sent"

    nltk.download("stopwords")
    *stopwords_ar, = set(stopwords.words('arabic'))

    if by != "sent":
      data[col] = [[word for word in sent if word not in stopwords_ar] for sent in data[col]]
    else:
      data[col] = [word for word in data[col] if word not in stopwords_ar]

    return data

def qalsadi_lemma(data, col="paragraph", by="sent"):
    lemmer = qalsadi.lemmatizer.Lemmatizer()

    if by != "sent":
      data[col] = [[lemmer.lemmatize(word) for word in sent] for sent in data[col]]
    else:
      data[col] = [lemmer.lemmatize(word) for word in data[col]]

    return data

def translate(data, col="paragraph", max_length=512, from_lang="ar", to_lang="en"):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(device)

    batches = list(generate_batch_sized_chunks(data[col], 16))
    model_name = f'Helsinki-NLP/opus-mt-{from_lang}-{to_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    summary = []

    for batches in tqdm(batches):
        tokenized_data = tokenizer(batches, padding=True, truncation=True, is_split_into_words=True if isinstance(data[col][0], list) else False, return_tensors="pt")

        translated = model.generate(
            input_ids=tokenized_data["input_ids"].to(device),
            attention_mask=tokenized_data["attention_mask"].to(device)
            )

        decoded_summaries = tokenizer.batch_decode(translated, skip_special_tokens=True)
        summary = summary + decoded_summaries

    # summary = ["summarize: " + doc for doc in summary] # only for mt5 models

    data[f"{col}_translated"] = summary

    del model
    torch.cuda.empty_cache()

    return data

def preprocess_function(data, col="paragraph", max_length=512):

    if "summary" in col:
        labels = summ_tokenizer(data[col], max_length=max_length, truncation=True, padding=True, is_split_into_words=True if isinstance(data[col][0], list) else False)
        data["labels"] = labels["input_ids"]
    else:
      para_tokenize = summ_tokenizer(data[col], max_length=max_length, truncation=True, padding=True, is_split_into_words=True if isinstance(data[col][0], list) else False)
      data["input_ids"] = para_tokenize["input_ids"]
      data["attention_mask"] = para_tokenize["attention_mask"]

    return data