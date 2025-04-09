# ALSU - Your Personal Programming AI Assistant

ALSU is a lightweight AI programming assistant that runs entirely on your MacBook Air M2. It's built on the TinyLlama model and designed to help with coding questions, providing explanations, and generating code examples.

## Features

- **100% Local** - Runs entirely on your MacBook, no data sent to external servers
- **Privacy-Focused** - All conversations stay on your device
- **Programming Expertise** - Specialized in helping with code and programming concepts
- **Customizable** - Can be fine-tuned to match your preferences

## Quick Start

1. **Activate your virtual environment**:
   ```bash
   cd llm-project
   source llm-env/bin/activate
   ```

2. **Run ALSU**:
   ```bash
   python llm.py
   ```

3. **Select a model** when prompted (option 1 - TinyLlama is recommended)

4. **Use the following system prompt** when asked:
   ```
   You are ALSU, a specialized programming AI assistant. You excel at helping with code in multiple languages, especially Python, JavaScript, and other popular languages. You provide concise, accurate explanations and practical code examples. When answering programming questions, you focus on best practices, efficient solutions, and clear documentation. You're friendly but direct, prioritizing technical accuracy and practical advice. If you don't know something, you'll acknowledge it rather than guessing. Always format code examples properly with appropriate syntax highlighting.
   ```

5. **Start chatting with ALSU!**

## System Requirements

- MacBook with Apple Silicon (M1/M2/M3)
- macOS 12.3 or later
- Python 3.8 or later
- 8GB RAM minimum (16GB recommended)
- 3GB free disk space for model storage

## Dependencies

ALSU requires the following Python packages:
```bash
pip install torch torchvision torchaudio transformers accelerate
```

## Example Conversations

Try asking ALSU questions like:

- "How do I read a CSV file in Python?"
- "Explain the difference between == and === in JavaScript"
- "What's the best way to handle errors in Python?"
- "Show me how to create a simple React component"
- "How do I use async/await in JavaScript?"

## Fine-Tuning ALSU

To make ALSU even better at programming or adapt it to your style, you can fine-tune the model. See the `fine-tuning.md` file for detailed instructions.

### Quick Fine-Tuning Steps:

1. Install additional requirements:
   ```bash
   pip install datasets peft bitsandbytes trl
   ```

2. Prepare your training data in `training_data/programming_examples.jsonl`

3. Run the fine-tuning script:
   ```bash
   python finetune.py
   ```

4. Update your model path in `llm.py` to use the fine-tuned model

## Troubleshooting

### Common Issues:

- **Out of memory errors**: Close other applications and try using a smaller model
- **Slow responses**: The first few responses may be slow as the model warms up
- **Warnings about truncation/temperature**: These are normal and don't affect functionality
- **Disconnected responses**: Try asking more specific questions with clear context

### Reset the Environment:

If you encounter persistent issues, try resetting:
```bash
deactivate
source llm-env/bin/activate
```

## Project Structure

- `llm.py` - Main script that runs ALSU
- `finetune_data.py` - Helper script to prepare fine-tuning data
- `finetune.py` - Script to fine-tune the model
- `training_data/` - Directory containing training examples
- `alsu-finetuned/` - Directory where fine-tuned model is saved

## Extending ALSU

ALSU can be extended in several ways:

1. **Add a web interface** using Gradio or Streamlit
2. **Connect to coding tools** like GitHub or VS Code
3. **Incorporate code analysis tools** for better suggestions
4. **Build a knowledge base** of common programming solutions

## License

This project uses the TinyLlama model which is available under the Apache 2.0 license.

## Acknowledgments

- TinyLlama by the TinyLlama team
- Hugging Face Transformers library
- PyTorch and torchvision
