import nltk
from transformers import pipeline

def split_text(text, max_words=512):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    split_points = [i for i, word in enumerate(words) if word.endswith(('.', '!', '?'))]
    if not split_points:
        return [text[:max_words]]
    
    best_split = 0
    for i, split_point in enumerate(split_points):
        if split_point <= max_words:
            best_split = i
        else:
            break
    
    split_index = split_points[best_split]
    return [text[:split_index + 1]] + split_text(text[split_index + 1:], max_words)



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
