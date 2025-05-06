# Voicera Emotion Detector

A simple Gradio web app for detecting sentiment (Positive/Negative) in text using the DistilBERT transformer model.

## 🔥 Features

- **Sentiment Analysis**: Uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english`.
- **History Tracking**: Displays a list of past predictions in the current session.
- **Clear Input**: Quickly reset the text box.
- **Clear History**: Wipe the session history.

## 💡 Example

```plaintext
Input: "Hello, you are a good guy." → Sentiment: POSITIVE (confidence: 1.0)
Input: "Oh Jesus! Is him!" → Sentiment: POSITIVE (confidence: 0.831)
Input: "LMAOOOOO" → Sentiment: NEGATIVE (confidence: 0.909)
```

## 🧠 How it works
The app uses a pre-trained DistilBERT model fine-tuned for binary sentiment classification (positive/negative). User input is processed through the model and displayed along with the prediction confidence. The app supports session-based history tracking and provides easy-to-use UI controls

##  Tech Stack
- Hugging Face Transformers
- Gradio
- Python 3.x

## How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```
