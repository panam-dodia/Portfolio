import os
os.environ.pop("CUDA_PATH", None)
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from LLama 2 model

def getLLama_response(input_text, no_words, blog_style):

    # LLama2 model

    llm=CTransformers(model='D:/Portfolio/views/projects/blog/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01,
                              'gpu_layers': 0})
    
    # Prompt Template

    template="""
    write a bloh for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
    """

    prompt=PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                          template=template)
    
    # Generate the response from the LLama 2 model

    response=llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs", 
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs")

input_text=st.text_input("Enter the Blog Topic")

# Creating 2 more columns for additional 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the Blog for', ('Researchers','Data Scientist','Common People'), index=0)

submit=st.button("Generate")

# Final response
if submit:
    st.write(getLLama_response(input_text, no_words, blog_style))