import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import gradio as gr

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def analyze(text):
    score = sia.polarity_scores(text)
    return score

iface = gr.Interface(fn=analyze, 
                     inputs="text", 
                     outputs="text", 
                     title="NLTK Sentiment App")

iface.launch()
