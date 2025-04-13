# designer.py
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO


class Designer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_menu_image(self, month, day):
        special_dates = [
            (12, 25),  # Christmas
            (2, 14),  # Valentine's Day
            (10, 31),  # Halloween
            # Add more special dates as you like!
        ]

        if (month, day) in special_dates:
            if month == 12:
                prompt = """
                Generate a festive Christmas-themed image for Scar's Bakery
                """
            elif month == 2:
                prompt = """
                Generate a romantic and inviting image for Scar's Bakery with a Valentine's Day theme. Include the bakery name "Scar's Bakery" in a lovely font. Feature a delightful display of heart-shaped baked goods such as cookies, cupcakes, and perhaps a small cake decorated with hearts or roses. Incorporate Valentine's Day elements like hearts, roses, ribbons, and soft, romantic colors (reds, pinks, whites). The bakery name should be clearly visible and the focus should be on the baked goods and the Valentine's Day atmosphere.
                """
            elif month == 10:
                prompt = """
                Generate a spooky Halloween-themed image for Scar's Bakery
                """
            # Add more conditions for other special dates
        elif month == 6:  # June - Summer month theme
            prompt = """
            Generate a bright and cheerful summer-themed image for Scar's Bakery...
            """
        elif month == 4:  # April - Spring month theme
            prompt = """
            Generate a cheerful spring-themed image for Scar's Bakery...
            """
        else:  # Default theme
            prompt = """
            Generate a general bakery-themed image for Scar's Bakery...
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
                image_file_path = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/media/scar_bakery_menu.png"  # Let's save it as a PNG
                image.save(image_file_path)
                print(f"Menu image saved to: {image_file_path}")

                break
            elif part.text is not None:
                print(f"Text response: {part.text}")

        return image_file_path

    def generate_special_dish_image(self, special_product):
        prompt = f"""
        Generate a high-quality image of a delicious {special_product} from Scar's Bakery. The {special_product} should be the clear focus of the image. Use a simple, clean background, perhaps a light-colored surface. Add a subtle, neutral element to enhance the presentation, such as a plain white plate or a simple piece of parchment paper underneath or next to the {special_product}. The lighting should be soft and inviting, highlighting the textures and details of the {special_product}. Aim for a professional food photography style that makes the {special_product} look appealing and delicious.
        """

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,  # Use the dynamic prompt here
            config=types.GenerateContentConfig(
                temperature=0.0, top_p=0.8, response_modalities=["Text", "Image"]
            ),
        )

        image_file_path = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image_file_path = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/media/scar_bakery_special.png"  # Save as a different file name
                image.save(image_file_path)
                print(f"Special dish image saved to: {image_file_path}")
                break
            elif part.text is not None:
                print(f"Text response: {part.text}")

        return image_file_path
