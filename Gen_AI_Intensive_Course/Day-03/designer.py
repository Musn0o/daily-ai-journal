# designer.py
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont  # Import PIL modules
from io import BytesIO
import base64
import os


class MenuDesigner:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_menu_image(self):
        prompt = """
        Generate an image for Scar's Bakery that evokes a sense of warmth and joy, inviting customers to come in and enjoy delicious treats. The image should prominently feature the bakery name "Scar's Bakery" in a playful, eye-catching font. It should also include a delightful assortment of freshly baked goods, such as loaves of bread, croissants, pastries, and cookies. The image should use a cheerful color palette of warm yellows, soft blues, and inviting browns to create a welcoming atmosphere. The overall style should be lighthearted and inviting, with a focus on showcasing the deliciousness and variety of the bakery's offerings. 
        """

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,  # Use the dynamic prompt here
            config=types.GenerateContentConfig(
                temperature=0.0, top_p=0.5, response_modalities=["Text", "Image"]
            ),
        )

        image_file_path = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image_file_path = "Gen_AI_Intensive_Course/Day-03/media/scar_bakery_menu.png"  # Let's save it as a PNG
                image.save(image_file_path)
                print(f"Menu image saved to: {image_file_path}")

                break
            elif part.text is not None:
                print(f"Text response: {part.text}")

        return image_file_path
