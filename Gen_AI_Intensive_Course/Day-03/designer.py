# # designer.py
# import datetime
# import os
# from google import genai
# from google.genai import types
# from PIL import Image
# from io import BytesIO


# class Designer:
#     def __init__(self):
#         GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
#         self.client = genai.Client(api_key=GEMINI_API_KEY)

#     def generate_welcome_image(self, month, occasion_name):
#         if occasion_name:
#             prompt = f"""Generate an image with title "Welcome to Scar's Bakery" and {occasion_name}-theme"""
#         else:
#             if month in (1, 2, 12, 11):  # Winter (including November as early winter)
#                 prompt = """
#                 Generate a cozy winter-themed image with title "Welcome to Scar's Bakery"
#                 """
#             elif month in (3, 4, 5):  # Spring
#                 prompt = """
#                 Generate a cheerful spring-themed image with title "Welcome to Scar's Bakery"
#                 """
#             elif month in (6, 7, 8):  # Summer
#                 prompt = """
#                 Generate a bright and cheerful summer-themed image with title "Welcome to Scar's Bakery"
#                 """
#             elif month in (9, 10):  # Autumn/Fall
#                 prompt = """
#                 Generate a warm autumn-themed image with title "Welcome to Scar's Bakery"
#                 """
#             else:  # Default (shouldn't ideally be reached with valid month)
#                 prompt = """
#                 Generate a general bakery-themed image with title "Welcome to Scar's Bakery"
#                 """

#         response = self.client.models.generate_content(
#             # Change the model name here:
#             model="gemini-2.0-flash-exp)",
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 temperature=0.0, top_p=0.5, response_modalities=["Text", "Image"]
#             ),
#         )

#         image_file_path = None
#         for part in response.candidates[0].content.parts:
#             if part.inline_data is not None:
#                 image = Image.open(BytesIO((part.inline_data.data)))
#                 image_file_path = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/media/scar_bakery_welcome.png"  # Let's save it as a PNG
#                 image.save(image_file_path)
#                 print(f"Menu image saved to: {image_file_path}")

#                 break
#             elif part.text is not None:
#                 print(f"Text response: {part.text}")

#         return image_file_path

#     def generate_special_dish_image(self, special_product):
#         prompt = f"""
#         Generate a high-quality image of a delicious {special_product} from Scar's Bakery. The {special_product} should be the clear focus of the image. Use a simple, clean background, perhaps a light-colored surface. Add a subtle, neutral element to enhance the presentation, such as a plain white plate or a simple piece of parchment paper underneath or next to the {special_product}. The lighting should be soft and inviting, highlighting the textures and details of the {special_product}. Aim for a professional food photography style that makes the {special_product} look appealing and delicious.
#         """

#         response = self.client.models.generate_content(
#             # Change the model name here:
#             model="gemini-2.0-flash-exp",
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 temperature=0.0, top_p=0.5, response_modalities=["Text", "Image"]
#             ),
#         )

#         image_file_path = None
#         for part in response.candidates[0].content.parts:
#             if part.inline_data is not None:
#                 image = Image.open(BytesIO((part.inline_data.data)))
#                 image_file_path = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/media/scar_bakery_special.png"  # Save as a different file name
#                 image.save(image_file_path)
#                 print(f"Special dish image saved to: {image_file_path}")
#                 break
#             elif part.text is not None:
#                 print(f"Text response: {part.text}")

#         return image_file_path


# now = datetime.datetime.now()
# current_month = now.month
# occasion = "Halloween"
# designer = Designer()
# welcome_image = designer.generate_welcome_image(current_month, "Halloween")
# if welcome_image:
#     print(welcome_image)

# # Make sure your GOOGLE_API_KEY environment variable is set
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable not set.")

# client = genai.Client(api_key=GOOGLE_API_KEY)

# print("Listing available models supporting 'generateContent':")
# for model in client.models.list():
#     print(f"- {model.name})")
