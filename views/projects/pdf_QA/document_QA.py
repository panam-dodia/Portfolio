import streamlit as st
import openai
import os
from PyPDF2 import PdfReader

# Function to extract text from PDF
def read_pdf_context(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        context = ""
        for page in pdf_reader.pages:
            context += page.extract_text() + "\n"
        return context
    except Exception as e:
        return f"An error occurred while reading PDF: {e}"

def answer_question_with_openai(question, context):
    """Answer question using OpenAI API"""
    # Get API key
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        return {
            "answer": "Please configure your OpenAI API key to use document Q&A.",
            "success": False,
            "source": "no_api_key"
        }
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Limit context length to avoid token limits
        max_context_length = 12000  # Leave room for question and response
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create QA prompt
        prompt = f"""Based on the following document content, please answer the question. If the answer cannot be found in the document, say "I cannot find this information in the provided document."

Document content:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document content. Be precise and only answer based on the information given."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "success": True,
            "source": "openai_api"
        }
        
    except openai.APIError as e:
        return {
            "answer": f"OpenAI API error: {str(e)}",
            "success": False,
            "source": "api_error"
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "success": False,
            "source": "general_error"
        }

def main():
    st.title("📄 PDF Question and Answer")
    st.write("Upload a PDF and ask questions about its content using AI-powered document analysis.")
    
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
        
        st.subheader("ℹ️ How it works")
        st.write("""
        1. Upload a PDF document
        2. Ask a question about the content
        3. AI analyzes the document and provides answers
        4. Get precise answers based on document content
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        - Ask specific questions for better results
        - Questions should be answerable from the document
        - Try different phrasings if you don't get good results
        - Shorter documents generally provide better accuracy
        """)
        
        st.sidebar.divider()
        st.sidebar.caption("v3.0.0 | OpenAI-powered Document Q&A")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Your PDF", type=["pdf"])
    
    # Question input
    question = st.text_area("Input Your Question", height=100, 
                           placeholder="What is the main topic of this document?")
    
    # Initialize session state
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""
    if 'answer_source' not in st.session_state:
        st.session_state.answer_source = ""
    if 'last_document' not in st.session_state:
        st.session_state.last_document = ""
    
    if uploaded_file is not None and question.strip() != "":
        if not openai_api_key:
            st.error("Please configure your OpenAI API key to use document Q&A.")
            return
            
        # Process button
        if st.button("Get Answer", type="primary"):
            with st.spinner("Processing document and generating answer..."):
                # Extract text from PDF
                context = read_pdf_context(uploaded_file)
                
                if context.startswith("An error occurred"):
                    st.error(context)
                else:
                    # Display document info
                    word_count = len(context.split())
                    char_count = len(context)
                    st.info(f"📄 Document processed: {word_count:,} words, {char_count:,} characters")
                    
                    # Get answer using OpenAI
                    result = answer_question_with_openai(question, context)
                    
                    if result["success"]:
                        st.session_state.last_answer = result["answer"]
                        st.session_state.answer_source = result["source"]
                        st.session_state.last_document = uploaded_file.name
                        
                        # Display answer
                        st.subheader("Answer:")
                        st.success(result["answer"])
                        
                        # Show source
                        if result["source"] == "openai_api":
                            st.info("🚀 Answer generated using OpenAI GPT")
                        
                        # Add follow-up question suggestions
                        st.subheader("💡 Try asking:")
                        follow_up_questions = [
                            "Can you summarize the main points?",
                            "What are the key conclusions?",
                            "Are there any recommendations mentioned?",
                            "What methodology was used?"
                        ]
                        
                        cols = st.columns(2)
                        for i, q in enumerate(follow_up_questions):
                            with cols[i % 2]:
                                if st.button(q, key=f"followup_{i}"):
                                    st.session_state.question_area = q
                                    st.rerun()
                    else:
                        st.error("Failed to generate answer. Please check your API configuration.")
                        st.write(result["answer"])
    
    # Show last answer if available
    elif st.session_state.last_answer:
        st.subheader("Last Answer:")
        st.write(f"**Document:** {st.session_state.last_document}")
        st.write(st.session_state.last_answer)
        
        if st.session_state.answer_source == "openai_api":
            st.caption("🚀 Generated using OpenAI GPT")
    
    # Instructions when no file is uploaded
    if uploaded_file is None:
        st.info("👆 Please upload a PDF file to get started")
        
        # Show example questions
        st.subheader("📝 Example Questions You Can Ask:")
        example_questions = [
            "What is the main topic of this document?",
            "Can you provide a summary of the key findings?",
            "What recommendations are mentioned?",
            "Who are the main authors or contributors?",
            "What methodology was used in this research?",
            "What are the main conclusions?",
            "Are there any statistical findings mentioned?",
            "What future work is suggested?"
        ]
        
        for q in example_questions:
            st.write(f"• {q}")
            
    elif question.strip() == "":
        st.info("💭 Please enter a question about the uploaded document")

if __name__ == "__main__":
    main()