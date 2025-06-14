project:
  name: "WhatsApp GPT-Style Chatbot"
  description: >
    A custom Transformer-based language model trained on WhatsApp chat exports.
    It mimics user conversation styles using GPT-style architecture and BPE tokenizer.

features:
  - "📱 WhatsApp Preprocessing: Parses exported .txt files from WhatsApp"
  - "🧼 Cleans system, media, and spam messages"
  - "📊 Groups consecutive messages by sender"
  - "🧠 Custom GPT Transformer: Multi-head attention, layer norm, residuals"
  - "🔤 Byte Pair Encoding (BPE) Tokenizer with special tokens"
  - "🗃 Fine-tuning data formatted as: '<|startoftext|>Sender<|separator|>Message<|endoftext|>'"

structure:
  dataset:
    - "WhatsApp Chat with XYZ.txt: Raw chat files"
  tokenizer:
    - "my_tokenizer.model"
    - "my_tokenizer.vocab"
  minbpe:
    - "base.py"
    - "basic.py"
  encoded:
    - "encoded_data.txt"
  notebook:
    - "chatbot.ipynb"
    - "fine_tuning.json"

setup:
  dependencies:
    - torch
    - pandas
    - regex
    - tiktoken
  install_commands:
    - "pip install torch pandas regex tiktoken"
    - "pip install -e ./minbpe/"

preprocessing:
  module: "preprocess.py"
  function: "preprocess(file_path: str) -> pd.DataFrame"
  description: >
    Cleans chat logs, removes noise, extracts timestamp, sender, and message.

format:
  template: "<|startoftext|>Sender<|separator|>Message<|endoftext|>"
  tokenizer: "BasicTokenizer (BPE from minbpe)"
  output_file: "fine_tuning.json"

model:
  name: "GPTLanguageModel"
  type: "Decoder-only Transformer"
  details:
    blocks: 4
    heads: 4
    embedding_size: 256
    vocab_size: 1244
    total_parameters: "3.86M"
  architecture:
    - "Token Embedding"
    - "Positional Embedding"
    - "Multi-head Self Attention"
    - "Feed-Forward Network"
    - "LayerNorm + Residual Connections"
    - "Final Linear Layer"

training:
  torch_compile: true
  device_support: "cuda | cpu"
  todo:
    - "Implement training loop with optimizer & scheduler"
    - "Add dataloader for encoded dataset"
    - "Track loss, accuracy during training"

generation:
  method: "model.generate(input_tensor, max_new_tokens)"
  example:
    prompt: "<|startoftext|>Didi Jio<|separator|>Hello"
    output: "Hi! How are you today?"

future_work:
  - "Add training loop"
  - "Streamlit interface for conversation"
  - "Deploy with FastAPI or Flask"
  - "Evaluation metrics (BLEU, perplexity)"
  - "Integrate multi-user chat support"

author:
  name: "Sanchit Pandey"
  license: "MIT"
  
