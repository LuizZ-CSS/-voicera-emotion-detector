import gradio as gr
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text, history):
    if not text:
        # Return previous output and history unchanged
        if history:
            history_display_text = "\n".join(history)
        else:
            history_display_text = ""
        return "", history, history_display_text

    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 3)
    output = f"Sentiment: {label} (confidence: {score})"
    updated_history = history + [f"Input: {text} → {output}"]  # avoid modifying history in-place
    history_display_text = "\n".join(updated_history)
    return output, updated_history, history_display_text

def clear_history():
    return "", [], ""  # clear output, state, and display text

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

    history_display = gr.Textbox(label="History (this session)", lines=8, interactive=False)

    submit_btn.click(
        predict_sentiment,
        inputs=[text_input, history],
        outputs=[output, history, history_display]
    )

    clear_input_btn.click(
        lambda: "",
        None,
        text_input
    )

    clear_history_btn.click(
        clear_history,
        None,
        [output, history, history_display]
    )

if __name__ == "__main__":
    demo.launch()
