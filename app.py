import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import requests
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import time

load_dotenv()

# Check for Hugging Face API token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.warning("Please add your Hugging Face API token to a .env file.")
    st.info("Create a file named .env in the same directory as app.py and add the following line:")
    st.code("HUGGINGFACEHUB_API_TOKEN='your_hugging_face_api_token_here'")
    st.stop()

@st.cache_resource
def get_image_to_text_pipeline():
    """
    Returns a cached image-to-text pipeline.
    """
    pipe = HuggingFacePipeline.from_model_id(
        model_id="Salesforce/blip-image-captioning-large",
        task="image-to-text",
    )
    return pipe

def image_to_text(image):
    """
    Generates a description for an image using a Hugging Face model.
    """
    pipe = get_image_to_text_pipeline()
    # The output is a list of dictionaries, each with a "generated_text" key
    results = pipe.run(images=image)
    return results[0]['generated_text']

@st.cache_resource
def get_text_to_image_pipeline():
    """
    Returns a cached text-to-image pipeline.
    """
    pipe = HuggingFacePipeline.from_model_id(
        model_id="stabilityai/stable-diffusion-2",
        task="text-to-image",
    )
    return pipe

def generate_image(prompt):
    """
    Generates an image from a prompt using a Hugging Face model.
    """
    pipe = get_text_to_image_pipeline()
    # The output of the pipeline is a list with a PIL Image
    return pipe.run(prompt)[0]

st.title("Image Generation with LangChain and Hugging Face")

st.header("Upload a reference image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

prompt = st.text_input("Enter a prompt for the new image")

if st.button("Generate Image"):
    if uploaded_file is not None and prompt:
        with st.spinner("Generating image..."):
            # Get image description
            image = Image.open(uploaded_file).convert("RGB")
            image_description = image_to_text(image)
            st.write("Image Description:", image_description)

            # Enhance the prompt
            template = "a photo of {user_prompt}, based on the following description: {image_desc}"
            prompt_template = PromptTemplate.from_template(template)
            final_prompt = prompt_template.format(user_prompt=prompt, image_desc=image_description)

            st.write("Final Prompt:", final_prompt)
            
            # Generate the new image
            generated_image = generate_image(final_prompt)
            
            if generated_image:
                st.image(generated_image, caption="Generated Image.", use_column_width=True)
            else:
                st.error("Failed to generate image.")
    else:
        st.warning("Please upload an image and enter a prompt.") 