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







#### FINE-TUNING
# Fine-Tuning ALSU

This guide explains how to fine-tune your ALSU programming assistant to improve its capabilities and customize its behavior to your preferences.

## Why Fine-Tune?

Fine-tuning provides several benefits:
- Makes ALSU better at specific programming languages or frameworks
- Teaches it your preferred coding style and conventions
- Improves response quality and relevance
- Enhances personality and conversational style

## Prerequisites

Ensure you have the necessary packages:

```bash
pip install datasets peft bitsandbytes accelerate trl
```

## Step 1: Prepare Training Data

Create a file called `finetune_data.py`:

```python
import json
import os

# Create directory for training data
os.makedirs("training_data", exist_ok=True)

# Function to create training examples
def create_training_file(examples, filename):
    with open(f"training_data/{filename}", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    print(f"Created {filename} with {len(examples)} examples")

# Programming Q&A examples - ADD YOUR OWN EXAMPLES HERE
programming_examples = [
    {
        "instruction": "How do I create a list in Python?",
        "input": "",
        "output": "In Python, you can create a list using square brackets []. Here's an example:\n\n```python\n# Empty list\nmy_list = []\n\n# List with items\nnumbers = [1, 2, 3, 4, 5]\n\n# List with different data types\nmixed = [1, 'hello', 3.14, True]\n\n# List using list() function\nanother_list = list((1, 2, 3))\n```\n\nYou can access items in a list by their index, which starts at 0:"
    },
    {
        "instruction": "Explain the difference between let and const in JavaScript",
        "input": "",
        "output": "In JavaScript, `let` and `const` are both block-scoped variables introduced in ES6, but they differ in mutability:\n\n```javascript\n// let - can be reassigned\nlet x = 10;\nx = 20; // This is valid\n\n// const - cannot be reassigned\nconst y = 30;\n// y = 40; // This would cause an error\n```\n\nKey differences:\n1. Variables declared with `let` can be reassigned.\n2. Variables declared with `const` cannot be reassigned after initialization.\n3. Both provide block scope unlike `var` which is function scoped.\n4. When using `const` with objects or arrays, the contents can still be modified:"
    },
    {
        "instruction": "How do I read a file in Python?",
        "input": "",
        "output": "Python offers several ways to read files. Here are the most common methods:\n\n```python\n# Method 1: Read entire file as a string\nwith open('filename.txt', 'r') as file:\n    content = file.read()\n    print(content)\n\n# Method 2: Read file line by line\nwith open('filename.txt', 'r') as file:\n    for line in file:\n        print(line.strip())  # strip() removes the newline character\n\n# Method 3: Read all lines into a list\nwith open('filename.txt', 'r') as file:\n    lines = file.readlines()\n    print(lines)\n```\n\nThe `with` statement automatically closes the file when you're done, which is the recommended approach."
    },
    {
        "instruction": "Write a function to calculate Fibonacci numbers in Python",
        "input": "",
        "output": "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <= 0:\n        return \"Input must be a positive integer\"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(3, n+1):\n            a, b = b, a + b\n        return b\n\n# Example usage\nfor i in range(1, 11):\n    print(f\"Fibonacci({i}) = {fibonacci(i)}\")\n```\n\nFor larger Fibonacci numbers, you might want to use memoization to improve performance:\n\n```python\ndef fibonacci_memo(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 0:\n        return \"Input must be a positive integer\"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)\n        return memo[n]\n```"
    },
    {
        "instruction": "How do I make an HTTP request in JavaScript?",
        "input": "",
        "output": "In modern JavaScript, you can make HTTP requests using the Fetch API or Axios library:\n\n```javascript\n// Using Fetch API (built into modern browsers)\nfetch('https://api.example.com/data')\n  .then(response => {\n    if (!response.ok) {\n      throw new Error('Network response was not ok');\n    }\n    return response.json();\n  })\n  .then(data => {\n    console.log(data);\n  })\n  .catch(error => {\n    console.error('There was a problem with the fetch operation:', error);\n  });\n\n// Using async/await with Fetch\nasync function fetchData() {\n  try {\n    const response = await fetch('https://api.example.com/data');\n    if (!response.ok) {\n      throw new Error('Network response was not ok');\n    }\n    const data = await response.json();\n    console.log(data);\n  } catch (error) {\n    console.error('Error fetching data:', error);\n  }\n}\n\n// Using Axios (requires importing the library)\n// npm install axios\nimport axios from 'axios';\n\naxios.get('https://api.example.com/data')\n  .then(response => {\n    console.log(response.data);\n  })\n  .catch(error => {\n    console.error('Error:', error);\n  });\n```\n\nFetch is built into modern browsers, while Axios provides a more feature-rich API with automatic JSON parsing and better error handling."
    }
    # Add more examples here to improve training
]

# Create the training file
create_training_file(programming_examples, "programming_examples.jsonl")

print("Data preparation complete!")
```

### Tips for Creating Good Training Examples:

1. **Include diverse programming languages** - Python, JavaScript, HTML/CSS, etc.
2. **Cover various difficulty levels** - From basics to more complex topics
3. **Follow a consistent format** - Clear questions and well-structured answers
4. **Include code examples** - With proper formatting and comments
5. **Reflect your preferred style** - Use the coding style you prefer

## Step 2: Create the Fine-Tuning Script

Create a file called `finetune.py`:

```python
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from trl import SFTTrainer

# Check for MPS (Apple Silicon) support
device_map = "mps" if torch.backends.mps.is_available() else "auto"
print(f"Using device: {device_map}")

# Load dataset
dataset = load_dataset('json', data_files='training_data/programming_examples.jsonl', split='train')
print(f"Loaded {len(dataset)} examples")

# Model configuration - using TinyLlama
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./alsu-finetuned"

# Special handling for Apple Silicon (MPS)
if device_map == "mps":
    print("Using MPS device - loading model in standard mode")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
else:
    # Configure 4-bit quantization for memory efficiency on other devices
    print("Using quantization for efficient training")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA (Low-Rank Adaptation) - efficient fine-tuning
if device_map != "mps":  # Skip for MPS as it needs modifications
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,               # Rank
        lora_alpha=32,      # Alpha parameter
        lora_dropout=0.05,  # Dropout probability
        bias="none",
        task_type="CAUSAL_LM", # Task type
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention modules to fine-tune
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, peft_config)

# Training arguments - adjusted for MacBook
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,             # Fewer epochs for faster training
    per_device_train_batch_size=2,  # Small batch size for memory constraints
    gradient_accumulation_steps=4,  # Increase if memory issues
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,                      # Mixed precision
    bf16=False,                     # Not supported on most hardware
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none",               # Disable wandb reporting
    optim="adamw_torch"
)

# Function to format the prompts correctly
def formatting_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"<|user|>\n{example['instruction'][i]}"
        if example['input'][i]:
            text += f"\n{example['input'][i]}"
        text += f"\n<|assistant|>\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,             # Reduced for memory constraints
    packing=False,                  # Disable packing for simpler training
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
trainer.save_model(output_dir)
print(f"Training complete! Model saved to {output_dir}")
```

## Step 3: Run the Fine-Tuning Process

Execute the following commands to prepare your data and run the fine-tuning:

```bash
# First, prepare your training data
python finetune_data.py

# Then run the fine-tuning (this may take several hours)
python finetune.py
```

## Step 4: Use Your Fine-Tuned Model

Update your `llm.py` to use the fine-tuned model:

```python
# Change the model selection code in your main script to use your fine-tuned model
if choice == "4":
    model_name = "./alsu-finetuned"  # Path to your fine-tuned model
elif choice == "2":
    model_name = "microsoft/phi-2"
# ...etc.
```

Add an option in your model selection menu:
```python
print("4. ALSU Fine-tuned Model (customized for programming)")
```

## Troubleshooting Fine-Tuning

### Memory Issues

If you encounter out-of-memory errors:

1. **Reduce batch size**:
   ```python
   per_device_train_batch_size=1,  # Smaller batch size
   gradient_accumulation_steps=8,  # Increase this to compensate
   ```

2. **Reduce sequence length**:
   ```python
   max_seq_length=256,  # Smaller context window
   ```

3. **Use fewer training examples** - Start with 5-10 examples and gradually add more

### Training Takes Too Long

If training is too slow on your MacBook:

1. **Reduce number of epochs**:
   ```python
   num_train_epochs=1,  # Just one pass through the data
   ```

2. **Set a maximum number of steps**:
   ```python
   max_steps=100,  # Limit training steps
   ```

3. **Use simpler examples** with shorter outputs

## Alternative: Simple Fine-Tuning Approach

If the above approach is too resource-intensive, you can use a simpler method:

1. Create a file called `simple_finetune.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load your examples
examples = [
    {"text": "<|user|>\nHow do I create a list in Python?\n<|assistant|>\nIn Python, you can create a list using square brackets []. Example: my_list = [1, 2, 3]"},
    {"text": "<|user|>\nExplain functions in JavaScript\n<|assistant|>\nFunctions in JavaScript are reusable blocks of code. Example: function greet(name) { return `Hello, ${name}!`; }"},
    # Add more examples
]

# Convert to dataset
df = pd.DataFrame(examples)
dataset = Dataset.from_pandas(df)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./simple-alsu",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model
trainer.train()

# Save model
trainer.save_model("./simple-alsu")
```

2. Run it:
```bash
python simple_finetune.py
```

This approach uses less memory but may not be as effective as the LoRA method.

## After Fine-Tuning

Once you've fine-tuned ALSU, it should:
- Provide more programming-focused responses
- Match your preferred style and formatting
- Give more accurate and concise code examples
- Feel more like a personal assistant than a generic model

Continue to add more training examples over time to improve its capabilities!
