<<<<<<< HEAD
# Offline Reasoning Assistant

An offline AI assistant powered by a local Large Language Model (LLM) that leverages **chain-of-thought (CoT) reasoning** for iterative problem analysis and solution generation. This project utilizes [Hugging Face Transformers](https://huggingface.co/transformers/), [Gradio](https://gradio.app/), and [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) to provide an interactive and efficient experience.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Local & Cached Model:**  
  Loads a quantized model from `./local_model` locally. If the model is not found, it automatically downloads it from the Hugging Face Hub and caches it for future use.

- **Chain-of-Thought Reasoning:**  
  Utilizes iterative reasoning to break down problems and refine solutions through multiple chain-of-thought iterations.

- **Interactive Web Interface:**  
  Provides a user-friendly interface via Gradio where you can input your query and view both the reasoning process and the final answer.

- **Optimized Performance:**  
  Supports 4-bit quantization (via BitsAndBytes) and GPU acceleration (if available) for faster inference.

## Getting Started

### Prerequisites

- **Python 3.8+**
- [PyTorch](https://pytorch.org/) (GPU support recommended for optimal performance)
- Other dependencies are listed in [`requirements.txt`](requirements.txt)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/offline-reasoning-assistant.git
   cd offline-reasoning-assistant
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Gradio interface by running:

```bash
python main.py
```

This will launch a local web server (by default at [http://0.0.0.0:7970](http://0.0.0.0:7970)). Open the URL in your browser to interact with the assistant.

## How It Works

1. **Model Loading:**  
   The assistant attempts to load a locally cached, quantized model from `./local_model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. If unavailable, it downloads the model from Hugging Face Hub and caches it locally.

2. **Chain-of-Thought Reasoning:**  
   For each query, the assistant iterates through multiple reasoning steps:
   - Analyzes the input prompt.
   - Breaks down the problem into logical components.
   - Generates intermediate responses.
   - Refines the context for subsequent iterations.
   
   The final output includes both detailed reasoning and a synthesized final answer.

3. **Interactive Interface:**  
   The Gradio UI consists of:
   - An input box for your question or problem statement.
   - A section displaying the reasoning steps.
   - A section highlighting the final answer.

## Customization

- **Model & Quantization Settings:**  
  Modify the model parameters, quantization configuration, or pipeline settings directly in `main.py` as needed.

- **Gradio Interface:**  
  Customize the UI layout and CSS styling within the `create_interface` function in `main.py`.

## Troubleshooting

- **Port Issues:**  
  If port `7970` is already in use, the application will attempt to automatically select an available port.

- **CUDA/CPU Compatibility:**  
  The app automatically detects if a CUDA-enabled GPU is available; otherwise, it will run on CPU (with potentially slower performance).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements. For significant changes, consider opening an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
```
=======
# offline-reasoning-assistant
Offline Reasoning Assistant is an offline AI assistant powered by a locally cached, quantized LLM. It utilizes chain-of-thought reasoning for iterative problem analysis and solution generation, with an interactive Gradio interface for easy user interaction.
>>>>>>> a4e4b060f0bc7a125fced3df45c89bfd80b88513
