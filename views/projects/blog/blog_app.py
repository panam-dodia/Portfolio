import os
import streamlit as st
import openai

# Set page configuration
st.set_page_config(
    page_title="Generate Blogs", 
    layout='centered',
    initial_sidebar_state='collapsed'
)

def getLLama_response(input_text, no_words, blog_style):
    """Generate blog using OpenAI API"""
    # Get API key
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        return {
            "content": "Please configure your OpenAI API key in Streamlit secrets to use blog generation.",
            "success": False,
            "source": "no_api_key"
        }
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Create prompt
        prompt = f"""Write a blog for {blog_style} job profile about the topic "{input_text}" in approximately {no_words} words.

The blog should be:
- Informative and engaging
- Tailored for {blog_style} audience
- Well-structured with clear sections
- Professional yet accessible

Topic: {input_text}
Target audience: {blog_style}
Word count: {no_words} words"""
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional blog writer who creates engaging, informative content tailored to specific audiences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=min(int(no_words) + 200, 2000),  # Add buffer but cap at 2000
            temperature=0.7
        )
        
        generated_content = response.choices[0].message.content.strip()
        
        return {
            "content": generated_content,
            "success": True,
            "source": "openai_api"
        }
        
    except openai.APIError as e:
        return {
            "content": f"OpenAI API error: {str(e)}",
            "success": False,
            "source": "api_error"
        }
    except Exception as e:
        return {
            "content": f"Error generating blog: {str(e)}",
            "success": False,
            "source": "general_error"
        }

def create_fallback_blog(input_text, no_words, blog_style):
    """Create a structured fallback blog when API is unavailable"""
    return f"""# {input_text.title()}

*A {blog_style.lower()}-focused perspective*

## Introduction

{input_text} represents a significant area of interest for {blog_style.lower()}. This blog explores the key aspects, implications, and opportunities within this domain.

## Key Concepts

Understanding {input_text} requires examining several fundamental components:

- **Definition and Scope**: What exactly constitutes {input_text} and its boundaries
- **Current Applications**: How {input_text} is being utilized today
- **Technical Considerations**: Important technical aspects relevant to {blog_style.lower()}
- **Best Practices**: Proven approaches and methodologies

## Practical Implications

For {blog_style.lower()}, {input_text} offers several practical benefits:

1. **Enhanced Understanding**: Deeper insights into the subject matter
2. **Improved Decision Making**: Better informed choices based on solid foundations
3. **Innovation Opportunities**: Potential for new approaches and solutions
4. **Professional Development**: Skills and knowledge advancement

## Future Outlook

The field of {input_text} continues to evolve, presenting new challenges and opportunities. {blog_style} should stay informed about:

- Emerging trends and developments
- New tools and technologies
- Evolving best practices
- Industry standards and regulations

## Conclusion

{input_text} remains a critical area for {blog_style.lower()} to understand and engage with. Continued learning and adaptation will be essential for success in this dynamic field.

---
*This blog was generated as a structured template. For more detailed, AI-generated content, please ensure your OpenAI API key is configured.*

**Word count**: Approximately {no_words} words"""

def main():
    st.header("Generate Blogs")
    
    # Get API key status
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Show API status
    if openai_api_key:
        st.success("🚀 OpenAI API: Configured")
    else:
        st.warning("⚠️ OpenAI API: Not configured (will use structured template)")
        with st.expander("How to configure OpenAI API"):
            st.write("""
            **For local development:**
            Set environment variable: `OPENAI_API_KEY=your_key_here`
            
            **For Azure deployment:**
            Add to Streamlit secrets or Azure App Settings:
            `OPENAI_API_KEY = "your_key_here"`
            """)
    
    # Input fields
    input_text = st.text_input("Enter the Blog Topic")
    
    # Creating 2 columns for additional fields
    col1, col2 = st.columns([5, 5])
    
    with col1:
        no_words = st.text_input('No of Words', value='200')
    with col2:
        blog_style = st.selectbox(
            'Writing the Blog for', 
            ('Researchers', 'Data Scientist', 'Common People'), 
            index=0
        )
    
    submit = st.button("Generate")
    
    # Initialize session state for generated content
    if 'generated_blog' not in st.session_state:
        st.session_state.generated_blog = ""
    if 'blog_source' not in st.session_state:
        st.session_state.blog_source = ""
    
    # Final response
    if submit:
        if input_text.strip() and no_words.strip():
            try:
                # Convert no_words to integer
                word_count = int(no_words)
                
                if word_count < 50 or word_count > 2000:
                    st.error("Please enter a word count between 50 and 2000.")
                    return
                
                with st.spinner("Generating blog content..."):
                    if openai_api_key:
                        # Use OpenAI API
                        result = getLLama_response(input_text, word_count, blog_style)
                    else:
                        # Use fallback template
                        content = create_fallback_blog(input_text, word_count, blog_style)
                        result = {
                            "content": content,
                            "success": True,
                            "source": "fallback_template"
                        }
                    
                    if result["success"]:
                        st.session_state.generated_blog = result["content"]
                        st.session_state.blog_source = result["source"]
                        
                        # Display the generated blog
                        st.subheader("Generated Blog:")
                        st.markdown(result["content"])
                        
                        # Show generation source
                        if result["source"] == "openai_api":
                            st.info("🚀 Generated using OpenAI GPT-3.5-turbo")
                        elif result["source"] == "fallback_template":
                            st.info("📝 Generated using structured template")
                        elif result["source"] == "no_api_key":
                            st.warning("🔑 API key required for AI generation")
                        
                        # Add download button for the blog
                        st.download_button(
                            label="📄 Download Blog",
                            data=result["content"],
                            file_name=f"blog_{input_text.replace(' ', '_').lower()}.md",
                            mime="text/markdown"
                        )
                        
                    else:
                        st.error("Blog generation failed. Using fallback template.")
                        fallback_content = create_fallback_blog(input_text, word_count, blog_style)
                        st.markdown(fallback_content)
                        
                        # Show fallback download
                        st.download_button(
                            label="📄 Download Fallback Blog",
                            data=fallback_content,
                            file_name=f"blog_{input_text.replace(' ', '_').lower()}_fallback.md",
                            mime="text/markdown"
                        )
                            
            except ValueError:
                st.error("Please enter a valid number for word count.")
            except Exception as e:
                st.error(f"Error generating blog: {str(e)}")
        else:
            st.warning("Please fill in both blog topic and word count.")
    
    # Show last generated blog if available
    elif st.session_state.generated_blog:
        st.subheader("Last Generated Blog:")
        st.markdown(st.session_state.generated_blog)
        
        # Show source
        if st.session_state.blog_source == "openai_api":
            st.caption("🚀 Generated using OpenAI GPT-3.5-turbo")
        elif st.session_state.blog_source == "fallback_template":
            st.caption("📝 Generated using structured template")
        elif st.session_state.blog_source == "no_api_key":
            st.caption("🔑 API key required for AI generation")
    
    # Add information in sidebar
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.write("""
        This blog generator uses OpenAI's GPT models to create content 
        tailored for different audiences and professions.
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        For best results:
        - Be specific with your blog topic
        - Choose appropriate word count (50-2000)
        - Select the right audience style
        - Review and edit the generated content
        """)
        
        st.subheader("🔧 API Configuration")
        if openai_api_key:
            st.success("OpenAI API configured")
        else:
            st.error("OpenAI API not configured")
            st.write("Set OPENAI_API_KEY in environment or secrets")
        
        st.sidebar.divider()
        st.sidebar.caption("v3.0.0 | OpenAI-powered Blog Generation")

if __name__ == "__main__":
    main()