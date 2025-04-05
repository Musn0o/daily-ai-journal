# from google import genai
# from google.genai import types
# from PIL import Image
# from io import BytesIO

# import PIL.Image

# image = PIL.Image.open("Gen_AI_Intensive_Course/Day-03/media/scar_bakery_menu.png")

# client = genai.Client()

# text_input = "Can you add a circle around the 'Croissant'?"

# response = client.models.generate_content(
#     model="gemini-2.0-flash-exp-image-generation",
#     contents=[text_input, image],
#     config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
# )

# for part in response.candidates[0].content.parts:
#     if part.text is not None:
#         print(part.text)
#     elif part.inline_data is not None:
#         image = Image.open(BytesIO(part.inline_data.data))
#         # image.show()
#         image_file_path = "Gen_AI_Intensive_Course/Day-03/media/updated_menu.png"  # Let's save it as a PNG
#         image.save(image_file_path)
# if image_file_path:
#     try:
#         img = Image.open(image_file_path)
#         draw = ImageDraw.Draw(img)
#         try:
#             font = ImageFont.truetype(
#                 "Gen_AI_Intensive_Course/Day-03/Fonts/bakery_font.ttf",
#                 size=20,
#             )
#         except IOError:
#             font = ImageFont.load_default()
#             print("Custom font not found, using default.")

#         image_width, image_height = img.size

#         # Calculate the starting Y-coordinate (around 40% from the top)
#         start_y = int(image_height * 0.30)
#         start_x = int(image_width * 0.10)
#         line_height = font.getbbox("A")[3]

#         current_y = start_y  # Use a variable to keep track of the current y position

#         for product, details in menu_data.items():
#             # Draw product name
#             draw.text(
#                 (start_x, current_y), product, fill=(0, 0, 0), font=font
#             )  # Black color
#             current_y += line_height + 2  # Move to the next line

#             # Draw price (slightly indented)
#             price_x = int(image_width * 0.15)  # Slightly more indented using percentage
#             draw.text(
#                 (price_x, current_y),
#                 f"${details['price']:.2f}",
#                 fill=(0, 128, 0),
#                 font=font,
#             )  # Green color for price
#             current_y += line_height + 2  # Move to the next line

#             # Draw description
#             description = details["description"]
#             draw.text(
#                 (start_x, current_y),
#                 description,
#                 fill=(50, 50, 50),
#                 font=font,
#             )  # Gray color
#             current_y += line_height + 10  # Add some extra space after the description

#         img.save(image_file_path)

#     except Exception as e:
#         print(f"Error opening or drawing on image: {e}")
#         return image_file_path  # Return the path even if drawing fails for now
