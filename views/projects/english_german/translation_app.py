import streamlit as st
import openai
import os

# Set page configuration
st.set_page_config(
    page_title="English to German Translator",
    page_icon="🌍",
    layout="wide"
)

def translate_with_openai(text, source_lang="English", target_lang="German"):
    """Translate text using OpenAI API"""
    # Get API key
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        return {
            "translation": "Please configure your OpenAI API key to use translation.",
            "success": False,
            "source": "no_api_key"
        }
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Create translation prompt
        prompt = f"Translate the following {source_lang} text to {target_lang}. Provide only the translation without any additional text:\n\n{text}"
        
        # Generate translation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate from {source_lang} to {target_lang} accurately and naturally."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        translation = response.choices[0].message.content.strip()
        
        return {
            "translation": translation,
            "success": True,
            "source": "openai_api"
        }
        
    except openai.APIError as e:
        return {
            "translation": f"OpenAI API error: {str(e)}",
            "success": False,
            "source": "api_error"
        }
    except Exception as e:
        return {
            "translation": f"Translation error: {str(e)}",
            "success": False,
            "source": "general_error"
        }

def main():
    st.title("🌍 English to German Translator")
    st.write("Translate English text to German using AI-powered translation.")
    
    # Get API key status
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Show API status in sidebar
    with st.sidebar:
        if openai_api_key:
            st.success("🚀 OpenAI API: Configured")
        else:
            st.error("⚠️ OpenAI API: Not configured")
            with st.expander("How to configure OpenAI API"):
                st.write("""
                **For local development:**
                Set environment variable: `OPENAI_API_KEY=your_key_here`
                
                **For Azure deployment:**
                Add to Azure App Settings:
                `OPENAI_API_KEY = "your_key_here"`
                """)
        
        st.subheader("ℹ️ About")
        st.write("""
        This translator uses OpenAI's GPT models to provide high-quality 
        English to German translations with natural language understanding.
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        For best results:
        - Use clear, grammatically correct English
        - Provide context for ambiguous terms
        - Review translations for important documents
        - Consider cultural nuances in translation
        """)
        
        # Add version info
        st.sidebar.divider()
        st.sidebar.caption("v3.0.0 | OpenAI-powered Translation")
    
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
            "Greeting": "Hello, how are you today?",
            "Business": "We need to schedule a meeting to discuss the project timeline.",
            "Travel": "I would like to book a hotel room for two nights.",
            "Technology": "Machine learning is transforming various industries."
        }
        
        cols = st.columns(len(examples))
        for col, (name, text) in zip(cols, examples.items()):
            if col.button(name, key=f"btn_{name.lower()}"):
                st.session_state.input_text_area = text
                st.rerun()
    
    with col2:
        st.subheader("German Translation")
        
        # Initialize session state for translation
        if 'translation' not in st.session_state:
            st.session_state.translation = ""
        if 'translation_source' not in st.session_state:
            st.session_state.translation_source = ""
            
        translation_placeholder = st.empty()
        translation_area = translation_placeholder.text_area(
            "Translation:",
            value=st.session_state.translation,
            height=200,
            disabled=True,
            key="output_text_area"
        )
        
        # Show translation source
        if st.session_state.translation_source:
            if st.session_state.translation_source == "openai_api":
                st.success("🚀 Translated using OpenAI GPT")
            elif st.session_state.translation_source == "no_api_key":
                st.warning("🔑 API key required for translation")
            elif st.session_state.translation_source == "api_error":
                st.error("❌ Translation API error")
    
    # Add translation button
    if st.button("Translate", type="primary", key="translate_button"):
        if input_text:
            if not openai_api_key:
                st.error("Please configure your OpenAI API key to use translation.")
                return
                
            try:
                with st.spinner("Translating..."):
                    result = translate_with_openai(input_text)
                    
                    if result["success"]:
                        st.session_state.translation = result["translation"]
                        st.session_state.translation_source = result["source"]
                        
                        # Update the translation area
                        translation_placeholder.text_area(
                            "Translation:",
                            value=result["translation"],
                            height=200,
                            disabled=True,
                            key="translation_result"
                        )
                        
                        # Show copy button
                        if st.button("📋 Copy Translation", key="copy_button"):
                            st.code(result["translation"])
                    else:
                        st.error("Translation failed. Please check your API configuration.")
                        st.write(result["translation"])
                        
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()