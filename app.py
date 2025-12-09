import streamlit as st
import json
from utils.prompt_generator import build_prompt, build_prompt_image, build_prompt_pancho
from utils.sd_api import generate_image_pancho, generate_image_sd, generate_style_image
from PIL import Image
import io
from utils.chatbot_rag import get_artist_answer, ARTIST_URLS


# Load style presets
with open("config/styles.json", "r") as f:
    styles = json.load(f)


def inic():
    st.header("ARTROOM AI")
    st.subheader("This is an interactive tool powered by IA, Streamlite and Langchain for learning about the greatest painters in history \n"
                 "For using this tool, select one of the rooms on the left and learn about the life and style of the greatest painters")
    



def chatbot_rag():

    st.header("Talk with the Artist (RAG Chatbot)")

    artist = st.selectbox(
        "Select the artist you like to talk with:",
        list(ARTIST_URLS.keys())
    )

    question = st.text_input("Enter your question for this artist:")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            output = get_artist_answer(artist, question)

        st.subheader("Answer:")
        st.write(output["Answer"])

        #st.subheader("Reference (text used):")
        #st.write(output["Reference"])

        #st.subheader("Sources:")
        #for src in output["Sources"]:
        #    st.markdown(f"- [{src}]({src})")



def text_to_image():
    st.header("Ask the artist to make a paiting for you (Text → Image )")
    
    #user text
    user_text = st.text_area("Describe the scene:", height=150)
    
    
    selected_style = st.selectbox(
        "Select the artist:", 
        list(styles.keys()),
        format_func=lambda x: styles[x]["name"]
    )
    
    style_prompt = styles[selected_style]["style_prompt"]
    
    if st.button("Generate Image"):
        if not user_text:
            st.warning("Please write a description.")
            return
        
        with st.spinner("Interpreting the description in my own style..."):
            final_prompt = build_prompt(user_text, style_prompt)
        
        st.subheader("Final Description")
        st.write(final_prompt)
        
        with st.spinner("Generating image with Stable Diffusion"):
            image_url = generate_image_sd(final_prompt)
        
        st.image(image_url, caption="Generated Image")


def image_to_image():
    st.header("Ask the artist to paint your image (Image → Image )")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    selected_style = st.selectbox("Choose an artist", list(styles.keys()))
    strength = st.slider("Style Strength", 0.1, 1.0, 0.65)

    if st.button("Generate Image"):
        if uploaded_image is None:
            st.error("Please upload an image first!")
            return

        # Convert to bytes
        img = Image.open(uploaded_image).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        st.subheader("Original Image")
        st.image(img)

        #Enhanced prompt
        style = styles[selected_style]["style_prompt"]
        final_prompt = build_prompt_image(style)

        with st.spinner("Drawing the picture..."):
            result_url = generate_style_image(
                init_image_bytes=img_bytes,
                style_prompt=final_prompt,
                strength=strength
            )

        st.subheader("Styled Result")
        st.image(result_url)



def pancho_fierro_experience():
    st.header("The Pancho Fierro Experience")
    st.markdown("#### This room generates images from text, using a model fine-tuned with 30 images of Pancho Fierro.")
    st.markdown("###### Pancho Fierro was an Afro-Peruvian watercolor painter who lived in Lima in the nineteenth century "
        "and depicted everyday characters from the city with mainly descriptive purpose. You can see some of his work here."
    )


    
    user_text = st.text_area("Describe a character to make Fierro to paint it:", height=150)
    
    
    
    if st.button("Generate Image"):
        if not user_text:
            st.warning("Please write a description.")
            return
        
        with st.spinner("Enhancing prompt"):
            final_prompt = build_prompt_pancho(user_text, "Pancho Fierro style")
        
        st.subheader("Final SD Prompt")
        
        with st.spinner("Drawing the picture..."):
            image_url = generate_image_pancho(final_prompt)
        
        st.image(image_url, caption="Generated Image")





def main():
    #st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to:", ("Home", "Chatbot", "Image generator", "Image converter", "Pancho Fierro Experience (fine-tuned model)"))

    if tab == "Home":
        inic()

    if tab == "Chatbot":
        chatbot_rag()

    if tab == "Image generator":
        text_to_image()

    if tab == "Image converter":
        image_to_image()
    
    if tab == "Pancho Fierro Experience (fine-tuned model)":
        pancho_fierro_experience()

if __name__ == "__main__":
    main()