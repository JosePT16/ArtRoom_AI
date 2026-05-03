import base64
import os

import replicate
from openai import OpenAI

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _dalle_size(width: int, height: int) -> str:
    if width > height:
        return "1792x1024"
    if height > width:
        return "1024x1792"
    return "1024x1024"


def generate_image_sd(prompt: str, width=768, height=768):

    if OPENAI_API_KEY is None:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=_dalle_size(width, height),
        quality="standard",
        n=1,
        response_format="url",
    )

    if response.data and response.data[0].url:
        return response.data[0].url

    raise ValueError(f"Unexpected OpenAI image response: {response}")


def generate_image_pancho(prompt: str, width=768, height=768):

    if REPLICATE_API_TOKEN is None:
        raise ValueError("Missing REPLICATE_API_TOKEN environment variable.")

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    model = "josept/pacho_fierro_style_266:9ce87cc42b9db4e8200b5447fa07f3e96e9a079980c05a9eb1814dac49c8d6a3"

    output = client.run(
        model,
        input={
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 50,
            "guidance": 10,
        },
    )

    if isinstance(output, list):
        first = output[0]
        if hasattr(first, "url"):
            return first.url
        if hasattr(first, "urls") and first.urls:
            return first.urls[0]

    if hasattr(output, "url"):
        return output.url

    if isinstance(output, str):
        return output

    raise ValueError(f"Unexpected output type: {type(output)}, content: {output}")


def generate_style_image(init_image_bytes, style_prompt, strength=0.65):

    client = replicate.Client()

    encoded = base64.b64encode(init_image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    response = client.run(
        "prunaai/p-image-edit",
        input={
            "prompt": style_prompt,
            "strength": strength,
            "images": [data_url],
            "turbo": True,
        },
    )

    if isinstance(response, list):
        out = response[0]
        if hasattr(out, "url"):
            return out.url
        return str(out)

    if hasattr(response, "url"):
        return response.url

    if isinstance(response, str):
        return response

    return str(response)
