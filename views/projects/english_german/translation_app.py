import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# Set page configuration
st.set_page_config(
    page_title="English to German Translator",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def load_translation_model():
    """Load the translation model and tokenizer"""
    try:
        # Use a pre-trained model from Hugging Face
        model_name = "Helsinki-NLP/opus-mt-en-de"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def translate_text(text, model, tokenizer, device):
    """Translate text from English to German"""
    # Tokenize input text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512
        )
    
    # Decode translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def main():
    st.title("🌍 English to German Translator")
    st.write("Translate English text to German using neural machine translation.")
    
    # Load model and tokenizer
    model, tokenizer = load_translation_model()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is not None:
        model = model.to(device)
    
    # Device info
    if torch.cuda.is_available():
        st.sidebar.success("🚀 Using GPU: " + torch.cuda.get_device_name(0))
    else:
        st.sidebar.info("💻 Using CPU")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("English Text")
        input_text = st.text_area(
            "Enter English text:",
            height=200,
            placeholder="Type or paste English text here...",
            key="input_text_area"
        )
        
        # Example buttons
        st.subheader("Try Examples:")
        examples = {
            "Greeting": "Hello, how are you?",
            "Learning": "I love learning German.",
            "Weather": "The weather is nice today.",
            "Meeting": "Let's meet tomorrow for coffee."
        }
        
        cols = st.columns(len(examples))
        for col, (name, text) in zip(cols, examples.items()):
            if col.button(name, key=f"btn_{name.lower()}"):
                input_text = text
                st.session_state.input_text = text
                st.rerun()
    
    with col2:
        st.subheader("German Translation")
        if 'translation' not in st.session_state:
            st.session_state.translation = ""
            
        translation_placeholder = st.empty()
        translation_area = translation_placeholder.text_area(
            "Translation:",
            value=st.session_state.translation,
            height=200,
            disabled=True,
            key="output_text_area"
        )
    
    # Add translation button
    if st.button("Translate", type="primary", key="translate_button"):
        if input_text:
            try:
                with st.spinner("Translating..."):
                    translation = translate_text(input_text, model, tokenizer, device)
                    st.session_state.translation = translation
                    translation_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=200,
                        disabled=True,
                        key="translation_result"
                    )
                    
                    # Add copy button
                    if st.button("📋 Copy Translation", key="copy_button"):
                        st.code(translation)
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")
    
    # Add information in the sidebar
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.write("""
        This translator uses a neural machine translation model based on the 
        Transformer architecture. It's fine-tuned on English-German parallel text.
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        For best results:
        - Use clear, grammatically correct English
        - Keep sentences at a reasonable length
        - Check translations of important documents with a native speaker
        """)
        
        # Add version info
        st.sidebar.divider()
        st.sidebar.caption("v1.0.0 | Neural Machine Translation")

if __name__ == "__main__":
    main()