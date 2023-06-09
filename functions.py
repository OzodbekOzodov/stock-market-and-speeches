# import nltk
from transformers import pipeline

import nltk

def split_text(text, max_words=512, max_seq_length=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk) + len(words) <= max_words:
            current_chunk += sentence + " "
        else:
            while words:
                if len(current_chunk.split()) + len(words) > max_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk += " ".join(words[:max_words - len(current_chunk.split())]) + " "
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                words = words[max_words:]
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    chunks = [chunk for chunk in chunks if len(chunk.split()) <= max_seq_length]
    
    return chunks




def calculate_sentiment_distilbert(text, classifier=None):
    """
    Calculate sentiment scores for a given text using a DistilBERT-based classifier.

    Parameters:
    text (str): The input text for sentiment analysis.
    classifier (pipeline, optional): The DistilBERT-based classifier. Defaults to None.

    Returns:
    tuple: A tuple containing the average positive and negative sentiment scores.

    """
    if classifier is None:
        classifier = pipeline(
            "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    chunks = split_text(text, max_words=512)
    sentiment_scores = []
    
    for chunk in chunks:
        sentiment = classifier(chunk)[0]
        sentiment_scores.append(sentiment['score'])
    
    if sentiment_scores:
        avg_positive = sum(sentiment_scores) / len(sentiment_scores)
        avg_negative = 1 - avg_positive
    else:
        avg_positive = 0.0
        avg_negative = 0.0
    
    return avg_positive, avg_negative


from transformers import BertForSequenceClassification, BertTokenizer
import torch

def calculate_sentiment_finbert(text, tokenizer=None, model=None):
    """
    Calculate sentiment scores for a given text using FinBERT.

    Parameters:
    text (str): The input text for sentiment analysis.
    tokenizer (BertTokenizer, optional): The FinBERT tokenizer. Defaults to None.
    model (BertForSequenceClassification, optional): The FinBERT model. Defaults to None.

    Returns:
    tuple: A tuple containing the average positive and negative sentiment scores.

    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

    if model is None:
        model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

    chunks = split_text(text, max_words=512)
    sentiment_scores = []
    max_words=512
    for chunk in chunks:
        encoded_input = tokenizer.encode_plus(
            chunk,
            add_special_tokens=True,
            truncation=True,
            max_length= max_words,
            padding='max_length',
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encoded_input)
            predictions = torch.softmax(outputs.logits, dim=1).tolist()[0]
        
        sentiment_scores.append(predictions)
    
    if sentiment_scores:
        avg_positive = sum(score[2] for score in sentiment_scores) / len(sentiment_scores)
        avg_negative = sum(score[0] for score in sentiment_scores) / len(sentiment_scores)
    else:
        avg_positive = 0.0
        avg_negative = 0.0
    
    return avg_positive, avg_negative
