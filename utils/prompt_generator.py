import openai

def build_prompt(user_text: str, style_prompt: str):
    """Use GPT to expand user description into a detailed art prompt."""
    
    system_msg = """
    You are a prompt engineer for Stable Diffusion.
    Expand the user prompt into a detailed visual description for an image generator.
    ALWAYS include composition, color, lighting, mood, and artistic elements.
    Combine with the target art style.
    """

    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"User description: {user_text}\nStyle: {style_prompt}"}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content



def build_prompt_pancho(user_text: str, style_prompt: str):
    """Use GPT to expand user description into a detailed art prompt."""
    
    system_msg = """
    You are a prompt engineer for Stable Diffusion.
    Correct the user prompt for giving it to the Pancho Fierro art style model.
    This implies only adding:
    - Draw the image as if it was a old watercolor painting in PANCHO_FIERRO_STYLE
    - Marfil background as old watercolor paper as PANCHO_FIERRO_STYLE
    - Use color, texture and brush strokes of PANCHO_FIERRO_STYLE
    - The composition typical is individuals in the center, full body, with no background.
    - Description intention of the action scene
    - IMPORTANT: DO NOT INCLUDE BACKGROUND IF NOT SPECIFIED IN THE USER PROMPT
    """

    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"User description: {user_text}\nStyle: {style_prompt}"}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def build_prompt_image(style_prompt: str):
    """Use GPT to create a detailed art prompt based on a style description."""
    
    system_msg =  """
    You are a prompt engineer for an image-to-image diffusion model.
    
    IMPORTANT:
    - The model MUST keep the structure, shapes, and layout of the input image.
    - The prompt SHOULD NOT tell the model to replace or redraw the image.
    - Do NOT use phrases like "turn this image into".
    - Instead use: "apply [style] artistic style while preserving the original content".
    - Focus on texture, brush strokes, color palette, lighting, mood.
    - DO NOT describe new objects or modify composition.
    - DO NOT generate a replacement.
    """

    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Apply this style: {style_prompt}"}

        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content