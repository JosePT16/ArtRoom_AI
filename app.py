import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.chatbot_rag import ARTIST_URLS, get_artist_answer
from utils.prompt_generator import build_prompt, build_prompt_image, build_prompt_pancho
from utils.sd_api import generate_image_pancho, generate_image_sd, generate_style_image


# Load style presets once at startup so the selected artist can drive every feature.
with open("config/styles.json", "r") as f:
    styles = json.load(f)


# Match the artist names used in the chatbot with the style keys used by the image features.
ARTIST_TO_STYLE_KEY = {
    "Van Gogh": "van_gogh",
    "Monet": "monet",
    "Picasso": "picasso",
    "Velasquez": "velazquez",
    "Dali": "dali",
    "Pancho Fierro": "fierro-no-fine-tunned",
}


# Map each artist to the image file shown on the right side of the home page.
ARTIST_TO_IMAGE = {
    "Van Gogh": Path("utils/pictures/Van Gogh.jpg"),
    "Monet": Path("utils/pictures/Monet.jpg"),
    "Picasso": Path("utils/pictures/Picasso.jpg"),
    "Velasquez": Path("utils/pictures/Velazquez.jpg"),
    "Dali": Path("utils/pictures/Dali.jpg"),
    "Pancho Fierro": Path("utils/pictures/Pancho Fierro.jpg"),
}


def home_page():
    # Use the home page as the main navigation flow for selecting an artist first.
    st.header("ARTROOM AI")
    st.subheader(
        "Choose an artist first, then decide whether you want to talk, paint, or redesign an image."
    )

    # Split the top of the home page so the controls stay on the left and the artist image sits on the right.
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        artist = st.selectbox(
            "Select an artist:",
            list(ARTIST_URLS.keys()),
        )

        # Store the selected artist so the chosen action can reuse it consistently.
        st.session_state.selected_artist = artist

        # Let the user choose the action after the artist has already been selected.
        action = st.radio(
            "What do you want to do?",
            ("Talk with the artist", "Paint something", "Redesign a picture"),
            horizontal=True,
        )

        # Resolve the matching visual style and render the active tool inside the left column.
        style_key = ARTIST_TO_STYLE_KEY[artist]

        if action == "Talk with the artist":
            chatbot_rag(artist)

        if action == "Paint something":
            text_to_image(artist, style_key)

        if action == "Redesign a picture":
            image_to_image(artist, style_key)

    with right_col:
        # Show the selected artist portrait when a matching local image exists.
        image_path = ARTIST_TO_IMAGE[artist]
        st.subheader(artist)
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.info(f"Add an image at `{image_path}` to show {artist} here.")


def chatbot_rag(artist):
    # Keep the selected artist visible inside the chatbot view.
    st.header(f"Talk with {artist}")

    # Keep chat history in session state so previous messages persist across reruns.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Reset the visible conversation when the user switches to a different artist.
    if st.session_state.get("chat_artist") != artist:
        st.session_state.chat_artist = artist
        st.session_state.chat_history = []

    # Use an art-related avatar for the assistant while keeping the default user icon.
    assistant_avatar = "🎨"

    # Render the stored conversation before accepting the next user message.
    for message in st.session_state.chat_history:
        avatar = assistant_avatar if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    question = st.chat_input("Enter your question for this artist:")

    if question:
        # Store the user's message immediately so it stays visible after Streamlit reruns.
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            output = get_artist_answer(artist, question)

        answer = output["Answer"]

        # Store and render the assistant reply so the page behaves like a conversation.
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(answer)

        # st.subheader("Reference (text used):")
        # st.write(output["Reference"])

        # st.subheader("Sources:")
        # for src in output["Sources"]:
        #     st.markdown(f"- [{src}]({src})")


def text_to_image(artist, style_key):
    # Reuse the selected artist from the home page instead of asking again here.
    st.subheader(f"Ask {artist} to paint something for you")

    # Keep the text prompt tied to the selected artist to avoid widget collisions.
    user_text = st.text_area("Describe the scene:", height=150, key=f"text_to_image_{style_key}")

    # Pull the style prompt from the artist selected on the home page.
    style_prompt = styles[style_key]["style_prompt"]

    if st.button("Generate Image", key=f"generate_text_to_image_{style_key}"):
        if not user_text:
            st.warning("Please write a description.")
            return

        # Route Pancho Fierro through his dedicated prompt builder and fine-tuned image model.
        if artist == "Pancho Fierro":
            with st.spinner("Esperame, ahora lo pinto"):
                final_prompt = build_prompt_pancho(user_text, "Pancho Fierro style")

            #st.subheader("Final Description")
            #st.write(final_prompt)

            with st.spinner("Generating image with the fine-tuned Pancho Fierro model..."):
                image_url = generate_image_pancho(final_prompt)
        else:
            with st.spinner("Interpreting the description in my own style..."):
                final_prompt = build_prompt(user_text, style_prompt)

            #st.subheader("Final Description")
            #st.write(final_prompt)

            with st.spinner("Generating image with Stable Diffusion"):
                image_url = generate_image_sd(final_prompt)

        st.image(image_url, caption="Generated Image")


def image_to_image(artist, style_key):
    # Reuse the selected artist from the home page instead of asking again here.
    st.subheader(f"Ask {artist} to redesign your picture")

    # Keep uploader widgets keyed by artist so Streamlit tracks them correctly.
    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key=f"image_to_image_upload_{style_key}",
    )
    strength = st.slider(
        "Style Strength",
        0.1,
        1.0,
        0.65,
        key=f"image_to_image_strength_{style_key}",
    )

    if st.button("Generate Image", key=f"generate_image_to_image_{style_key}"):
        if uploaded_image is None:
            st.error("Please upload an image first!")
            return

        # Convert the uploaded image into bytes before sending it to the model.
        img = Image.open(uploaded_image).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        st.subheader("Original Image")
        st.image(img)

        # Build the transformation prompt from the selected artist style.
        style = styles[style_key]["style_prompt"]
        final_prompt = build_prompt_image(style)

        with st.spinner("Drawing the picture..."):
            result_url = generate_style_image(
                init_image_bytes=img_bytes,
                style_prompt=final_prompt,
                strength=strength,
            )

        st.subheader("Styled Result")
        st.image(result_url)


def main():
    # Simplify the app into a single artist-first home page.
    home_page()


if __name__ == "__main__":
    main()
