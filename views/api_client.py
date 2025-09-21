import requests
import streamlit as st
import os
from typing import Optional

# API configuration
API_BASE_URL = os.getenv('MODEL_API_URL', 'http://localhost:8000')


class ModelAPIClient:
    @staticmethod
    def check_api_health() -> bool:
        """Check if API is available"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    # ---------------- Translation ----------------
    @staticmethod
    def translate_text(text: str, max_length: int = 512) -> dict:
        """Call translation API with fallback"""
        try:
            if not ModelAPIClient.check_api_health():
                return ModelAPIClient._fallback_translation(text)

            response = requests.post(
                f"{API_BASE_URL}/translate",
                json={"text": text, "max_length": max_length},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                return {
                    "translation": result["translation"],
                    "success": True,
                    "source": "api"
                }
            else:
                return ModelAPIClient._fallback_translation(text)

        except Exception as e:
            st.warning(f"API unavailable, using fallback: {str(e)}")
            return ModelAPIClient._fallback_translation(text)

    # ---------------- Blog Generation ----------------
    @staticmethod
    def generate_blog(input_text: str, no_words: int = 200, blog_style: str = "Common People") -> dict:
        """Call blog generation API with fallback"""
        try:
            if not ModelAPIClient.check_api_health():
                return ModelAPIClient._fallback_blog(input_text, no_words, blog_style)

            response = requests.post(
                f"{API_BASE_URL}/blog",
                json={
                    "input_text": input_text,
                    "no_words": no_words,
                    "blog_style": blog_style
                },
                timeout=60  # Blog generation takes longer
            )
            response.raise_for_status()
            result = response.json()

            return {
                "content": result.get("content", "Blog generation failed"),
                "success": result.get("success", False),
                "source": "api" if result.get("success") else "api_fallback"
            }

        except Exception as e:
            st.warning(f"API unavailable, using fallback: {str(e)}")
            return ModelAPIClient._fallback_blog(input_text, no_words, blog_style)

    # ---------------- QA ----------------
    @staticmethod
    def qa(question: str, context: str) -> dict:
        """Call QA API with fallback"""
        try:
            if not ModelAPIClient.check_api_health():
                return ModelAPIClient._fallback_qa(question, context)

            response = requests.post(
                f"{API_BASE_URL}/qa",
                json={"question": question, "context": context},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                return {
                    "answer": result["answer"],
                    "confidence": result.get("confidence", 0),
                    "success": True,
                    "source": "api"
                }
            else:
                return ModelAPIClient._fallback_qa(question, context)

        except Exception as e:
            st.warning(f"API unavailable, using fallback: {str(e)}")
            return ModelAPIClient._fallback_qa(question, context)

    # ---------------- Fallback Loaders ----------------
    @staticmethod
    @st.cache_resource
    def _load_fallback_translation_model():
        """Load translation model locally as fallback"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch

            model_name = "Helsinki-NLP/opus-mt-en-de"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            return model, tokenizer, device
        except Exception as e:
            st.error(f"Could not load fallback translation model: {e}")
            return None, None, None

    @staticmethod
    @st.cache_resource
    def _load_fallback_qa_pipeline():
        """Load QA pipeline locally as fallback"""
        try:
            from transformers import pipeline
            return pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            st.error(f"Could not load fallback QA model: {e}")
            return None

    # ---------------- Fallback Implementations ----------------
    @staticmethod
    def _fallback_translation(text: str) -> dict:
        """Fallback to local translation model"""
        model, tokenizer, device = ModelAPIClient._load_fallback_translation_model()

        if model is None:
            return {
                "translation": "Translation service unavailable",
                "success": False,
                "source": "fallback_failed"
            }

        try:
            import torch
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512
                )

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "translation": translation,
                "success": True,
                "source": "fallback_local"
            }

        except Exception as e:
            return {
                "translation": f"Translation failed: {str(e)}",
                "success": False,
                "source": "fallback_failed"
            }

    @staticmethod
    def _fallback_qa(question: str, context: str) -> dict:
        """Fallback to local QA model"""
        qa_pipeline = ModelAPIClient._load_fallback_qa_pipeline()

        if qa_pipeline is None:
            return {
                "answer": "QA service unavailable",
                "confidence": 0,
                "success": False,
                "source": "fallback_failed"
            }

        try:
            # Limit context length
            max_context_length = 4000
            if len(context) > max_context_length:
                context = context[:max_context_length]

            result = qa_pipeline(question=question, context=context)

            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "success": True,
                "source": "fallback_local"
            }

        except Exception as e:
            return {
                "answer": f"QA failed: {str(e)}",
                "confidence": 0,
                "success": False,
                "source": "fallback_failed"
            }

    @staticmethod
    def _fallback_blog(input_text: str, no_words: int, blog_style: str) -> dict:
        """Fallback blog generation (placeholder)"""
        fallback_content = f"""# {input_text}

This is a sample {blog_style.lower()} blog post about {input_text}. 

In approximately {no_words} words, this topic covers various aspects that would be relevant to {blog_style.lower()}. The blog generation service is currently unavailable, so this is a placeholder response.

Key points to consider:
- Understanding the fundamentals of {input_text}
- Practical applications and use cases
- Benefits and considerations
- Next steps for implementation

For a complete blog post, please ensure the model API service is running with the required LLama model file."""

        return {
            "content": fallback_content,
            "success": False,
            "source": "fallback_placeholder"
        }