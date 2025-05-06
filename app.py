import gradio as gr
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text, history):
    if not text:
        return "", history

    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 3)
    output = f"Sentiment: {label} (confidence: {score})"
    history.append(f"Input: {text} → {output}")
    return output, history

def clear_history():
    return [], []

with gr.Blocks() as demo:
    gr.Markdown("# Voicera Emotion Detector")
    gr.Markdown("Enter text to detect sentiment (Positive/Negative) using DistilBERT.")

    history = gr.State([])

    with gr.Row():
        text_input = gr.Textbox(lines=3, placeholder="Enter your text here...")
        output = gr.Textbox(label="Output")

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_input_btn = gr.Button("Clear Input")
        clear_history_btn = gr.Button("Clear History")

    history_display = gr.Textbox(label="History (this session)", lines=6, interactive=False)

    submit_btn.click(
        predict_sentiment,
        inputs=[text_input, history],
        outputs=[output, history_display]
    )

    clear_input_btn.click(
        lambda: "",
        None,
        text_input
    )

    clear_history_btn.click(
        clear_history,
        None,
        [history_display, history]
    )

if __name__ == "__main__":
    demo.launch()
