import os
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_PATH = os.path.join("./local_model", MODEL_NAME)


class Agent:
    def __init__(self, model_name: str = MODEL_NAME, model_path: str = MODEL_PATH):
        self.model_name = model_name
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

        # Initialize model components
        self.model, self.tokenizer = self._load_model()
        self.pipe = self._create_pipeline()
        self.logger.info("Agent initialization complete")

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with 4-bit quantization, handle download if missing"""
        try:
            return self._load_local_model()
        except Exception as e:
            self.logger.warning(f"Local model load failed: {e}")
            return self._download_and_cache_model()

    def _load_local_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Attempt to load locally cached model"""
        self.logger.info(f"Loading local model from {self.model_path}")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quant_config,
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
        """Download and cache model from Hugging Face Hub"""
        self.logger.info(f"Downloading {self.model_name} from Hugging Face Hub")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

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
        """Create text generation pipeline with optimized settings"""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    def generate_with_cot(self, prompt: str, iterations: int = 4) -> str:
        """Chain-of-thought reasoning with iterative refinement"""
        conversation_history = []
        current_context = prompt

        for iteration in range(1, iterations + 1):
            self.logger.info(f"CoT Iteration {iteration}/{iterations}")

            response = self._generate_step(current_context, iteration)
            cleaned_response = self._clean_response(response)

            conversation_history.append(
                f"## Iteration {iteration}:\n{cleaned_response}"
            )
            current_context = self._update_context(current_context, cleaned_response)

        return "\n\n".join(conversation_history)

    def _generate_step(self, context: str, iteration: int) -> str:
        """Generate a single CoT step"""
        system_prompt = """[INST] You are an AI assistant that uses chain-of-thought reasoning.
        Follow these steps:
        1. Carefully analyze the problem
        2. Break it into logical components
        3. Solve each component systematically
        4. Synthesize a comprehensive answer based on this chain-of-thoughts[/INST]"""

        prompt = f"{system_prompt}\n\nUser: {context}\nAssistant: Let's think step by step."

        return self.pipe(
            prompt,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]['generated_text']

    def _clean_response(self, response: str) -> str:
        """Clean and extract the assistant's response"""
        return response.split("Assistant:")[-1].strip()

    def _update_context(self, previous_context: str, response: str) -> str:
        """Update context for next iteration"""
        return f"{previous_context}\n\nPrevious analysis:\n{response}\nHow can we improve this?"


def create_interface():
    agent = Agent()

    def chat_interface(message: str) -> tuple[str, str]:
        """Gradio interface function with error handling"""
        try:
            process = agent.generate_with_cot(message)

            # Extract final answer (last iteration's response)
            final_answer = process.split("## Iteration")[-1].split("\n", 1)[-1].strip()
            return process, final_answer

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Error: {str(e)}", ""

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", font=[gr.themes.GoogleFont("Inter")])) as interface:
        gr.Markdown("""
        # 🧠 Offline Reasoning Assistant
        *Chain-of-Thought AI Powered by LOCAL LLM*
        """)

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
                        ["Explain quantum computing in simple terms"],
                        ["How to solve a quadratic equation?"],
                        ["What are the main causes of climate change?"]
                    ],
                    inputs=input_box,
                    label="Example Questions"
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Reasoning Process")
                process_view = gr.Markdown(
                    label="Thinking Steps",
                    value="*Processing steps will appear here...*",
                    elem_classes="process-box"
                )

                gr.Markdown("### 💡 Final Answer")
                answer_view = gr.Markdown(
                    label="Final Answer",
                    value="*Final answer will appear here...*",
                    elem_classes="answer-box"
                )

        submit_btn.click(
            fn=chat_interface,
            inputs=input_box,
            outputs=[process_view, answer_view]
        )

    # Add custom CSS
    interface.css = """
    .input-box textarea { font-size: 16px !important; padding: 15px !important; }
    .process-box { background: #1b1c1b; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; }
    .answer-box { background: #1d3e19; padding: 25px; border-radius: 10px; border: 2px solid #0d6efd; }
    .process-box markdown { color: #495057; line-height: 1.6; }
    .answer-box markdown { font-size: 18px; color: #0a58ca; }
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
            server_port=0,  # Auto-select port
            show_error=True
        )
