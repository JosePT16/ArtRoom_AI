import os
import replicate
import base64

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_image_sd(prompt: str, width=768, height=768):

    if REPLICATE_API_TOKEN is None:
        raise ValueError("Missing REPLICATE_API_TOKEN environment variable.")

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    model = "black-forest-labs/flux-1.1-pro"

    output = client.run(
        model,
        input={
            "prompt": prompt,
            "width": width,
            "height": height
        }
    )

    # CASE 1 — FileOutput with .url
    if hasattr(output, "url"):
        return output.url

    # CASE 2 — FileOutput with .urls (some models use plural)
    #if hasattr(output, "urls"):
    #    return output.urls[0]

    # CASE 3 — string
    #if isinstance(output, str):#       return output

    ## CASE 4 — list
    #if isinstance(output, list):
    #    return output[0]

    # CASE 5 — unexpected
    #raise ValueError(f"Unexpected output type: {type(output)}, content: {output}")


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
            "guidance": 10
        }
    )

    # Case: list containing FileOutput
    if isinstance(output, list):
        first = output[0]
        if hasattr(first, "url"):
            return first.url
        if hasattr(first, "urls") and first.urls:
            return first.urls[0]

    # Case: direct FileOutput
    if hasattr(output, "url"):
        return output.url

    # Case: fallback for edge scenarios
    if isinstance(output, str):
        return output

    raise ValueError(f"Unexpected output type: {type(output)}, content: {output}")


def generate_style_image(init_image_bytes, style_prompt, strength=0.65):

    client = replicate.Client()

    # Convert bytes to base64 data URL
    encoded = base64.b64encode(init_image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    response = client.run(
        "prunaai/p-image-edit",
        input={
            "prompt": style_prompt,
            "strength": strength,
            "images": [data_url],
            "turbo": True
        }
    )

    # if a list
    if isinstance(response, list):
        out = response[0]
        if hasattr(out, "url"): 
            return out.url
        return str(out)

    # if Single FileOutput object
    if hasattr(response, "url"):
        return response.url

    #if string
    if isinstance(response, str):
        return response

    # unexpected
    return str(response)
