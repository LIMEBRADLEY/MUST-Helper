from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

DEFAULT_CKPT_PATH = "/root/autodl-tmp/MUST-Helper"
PORT = 6006  # 在autodl上通常放开6006端口

app = Flask(__name__)

# Load model and tokenizer when the server starts
tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_CKPT_PATH,
    resume_download=True,
)

device_map = "auto" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_CKPT_PATH,
    torch_dtype="auto",
    device_map=device_map,
    resume_download=True,
).eval()
model.generation_config.max_new_tokens = 2048  # For chat.

def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
    return generated_text

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    history = data.get("history", [])

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        response = _chat_stream(model, tokenizer, query, history)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
