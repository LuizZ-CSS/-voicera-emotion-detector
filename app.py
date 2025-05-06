import gradio as gr
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 3)
    return f"Sentiment: {label} (confidence: {score})"

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."),
    outputs="text",
    title="Voicera Emotion Detector",
    description="Enter text to detect sentiment (Positive/Negative) using DistilBERT."
)

if __name__ == "__main__":
    iface.launch()
