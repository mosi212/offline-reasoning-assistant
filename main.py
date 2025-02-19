import os
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import transformers  # Import the entire module
import threading
import logging
from typing import Tuple, Generator

# Configure logging with debug level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_NAME = "Qwen/Qwen2.5-7B"
MODEL_PATH = os.path.join("../local_model", MODEL_NAME)

class Agent:
    def __init__(self, model_name: str = MODEL_NAME, model_path: str = MODEL_PATH):
        self.model_name = model_name
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

        # Combined chain-of-thought prompt template
        self.cot_template = (
            "Problem: {question}\n\n"
            "Step 1: Analysis - Identify key components and initial parameters.\n"
            "Assistant: <think> Provide a detailed analysis of the problem.\n\n"
            "Step 2: Planning - Outline the solution steps and describe the approach.\n"
            "Assistant: <think> Describe your solution plan in detail.\n\n"
            "Step 3: Execution - Perform the calculations step-by-step and verify intermediate results.\n"
            "Assistant: <think> Show all calculations and verify your results.\n\n"
            "Step 4: Synthesis - Synthesize a concise final answer and reflect on consistency.\n"
            "Assistant: Provide the final answer below."
        )

        self.logger.info("Initializing model components...")
        self.model, self.tokenizer = self._load_model()
        # self.model = torch.compile(self.model)
        self.pipe = self._create_pipeline()  # Now uses transformers.pipeline
        self.logger.info("Agent initialization complete")

    def _get_quant_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            return self._load_local_model()
        except Exception as e:
            self.logger.warning(f"Local model load failed: {e}. Downloading and caching model.")
            return self._download_and_cache_model()

    def _load_local_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.logger.info(f"Loading local model from {self.model_path}")
        quant_config = self._get_quant_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.logger.info("✅ Local model loaded successfully")
        return model, tokenizer

    def _download_and_cache_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.logger.info(f"Downloading {self.model_name} from Hugging Face Hub")
        quant_config = self._get_quant_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        os.makedirs(self.model_path, exist_ok=True)
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)
        self.logger.info(f"💾 Model cached at {self.model_path}")
        return model, tokenizer

    def _create_pipeline(self):
        """Create a text generation pipeline using transformers.pipeline."""
        return transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1
        )

    def _stream_generate_response(self, prompt: str, max_tokens: int) -> Generator[str, None, None]:
        self.logger.info("Starting streaming generation.")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "num_beams": 1,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ""
        for new_text in streamer:
            response += new_text
            yield response
        thread.join()
        self.logger.info("Streaming generation complete.")

    def generate_with_cot(self, prompt: str) -> Generator[str, None, None]:
        original_question = prompt.strip()
        combined_prompt = self.cot_template.format(question=original_question)
        self.logger.info(f"Combined prompt:\n{combined_prompt}")
        yield from self._stream_generate_response(combined_prompt, max_tokens=4096)

def create_interface():
    agent = Agent()

    def chat_interface(message: str) -> Generator[str, None, None]:
        try:
            yield from agent.generate_with_cot(message)
        except Exception as e:
            logger.exception("Generation error:")
            yield f"❌ Error: {str(e)}"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", font=[gr.themes.GoogleFont("Inter")])) as interface:
        gr.Markdown(
            """
            # 🧠 Offline Reasoning Assistant
            *Single-Iteration Streaming Chain-of-Thought AI Powered by LOCAL LLM*
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                input_box = gr.Textbox(
                    lines=4,
                    placeholder="Enter your question or problem statement...",
                    label="Your Question",
                    elem_classes="input-box"
                )
                submit_btn = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    examples=[
                        ["Solve for x in the quadratic equation: 2x^2 - 8x + 6 = 0"],
                        ["A rectangle has a length 4 times its width and an area of 64. Find its dimensions."],
                        ["Explain quantum computing in simple terms"]
                    ],
                    inputs=input_box,
                    label="Example Questions"
                )
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Output")
                output_view = gr.Markdown(
                    label="Output",
                    value="*The chain-of-thought reasoning will stream here...*",
                    elem_classes="process-box"
                )
        submit_btn.click(fn=chat_interface, inputs=input_box, outputs=output_view)
    interface.css = """
    .input-box textarea { font-size: 16px !important; padding: 15px !important; }
    .process-box { background: #1b1c1b; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; }
    footer { display: none !important; }
    """
    return interface

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7970,
            show_error=True,
            share=False
        )
    except OSError as e:
        logger.error(f"Port error: {e}")
        interface.launch(
            server_name="0.0.0.0",
            server_port=0,
            show_error=True
        )
