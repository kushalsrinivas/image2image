# Image Generation with LangChain and Hugging Face

This Streamlit application allows you to upload a reference image, provide a prompt, and generate a new image inspired by the reference. It uses a LangChain pipeline to enhance the input prompt and a Hugging Face model to generate the image.

## How to Run

1.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your Hugging Face API token:**

    Create a file named `.env` in the root of the project and add your Hugging Face API token to it:

    ```
    HUGGINGFACEHUB_API_TOKEN='your_hugging_face_api_token_here'
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    The application will open in your browser.

## How it Works

1.  **Upload Image and Prompt:** You upload a reference image and enter a text prompt.
2.  **Image Description:** An image-to-text model from Hugging Face (`Salesforce/blip-image-captioning-large`) generates a description of the uploaded image.
3.  **Prompt Enhancement:** A LangChain prompt template combines your original prompt with the generated image description to create a more detailed and context-aware prompt.
4.  **Image Generation:** A text-to-image model from Hugging Face (`stabilityai/stable-diffusion-2`) uses the enhanced prompt to generate a new image.
5.  **Caching:** The Hugging Face pipelines are cached using Streamlit's `@st.cache_resource` to ensure faster performance on subsequent runs.
# image2image
