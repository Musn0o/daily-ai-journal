# # from google import genai
# # from google.genai import types
# # from PIL import Image
# # from io import BytesIO

# # import PIL.Image

# # image = PIL.Image.open("Gen_AI_Intensive_Course/Day-03/media/scar_bakery_menu.png")

# # client = genai.Client()

# # text_input = "Can you add a circle around the 'Croissant'?"

# # response = client.models.generate_content(
# #     model="gemini-2.0-flash-exp-image-generation",
# #     contents=[text_input, image],
# #     config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
# # )

# # for part in response.candidates[0].content.parts:
# #     if part.text is not None:
# #         print(part.text)
# #     elif part.inline_data is not None:
# #         image = Image.open(BytesIO(part.inline_data.data))
# #         # image.show()
# #         image_file_path = "Gen_AI_Intensive_Course/Day-03/media/updated_menu.png"  # Let's save it as a PNG
# #         image.save(image_file_path)
# # if image_file_path:
# #     try:
# #         img = Image.open(image_file_path)
# #         draw = ImageDraw.Draw(img)
# #         try:
# #             font = ImageFont.truetype(
# #                 "Gen_AI_Intensive_Course/Day-03/Fonts/bakery_font.ttf",
# #                 size=20,
# #             )
# #         except IOError:
# #             font = ImageFont.load_default()
# #             print("Custom font not found, using default.")

# #         image_width, image_height = img.size

# #         # Calculate the starting Y-coordinate (around 40% from the top)
# #         start_y = int(image_height * 0.30)
# #         start_x = int(image_width * 0.10)
# #         line_height = font.getbbox("A")[3]

# #         current_y = start_y  # Use a variable to keep track of the current y position

# #         for product, details in menu_data.items():
# #             # Draw product name
# #             draw.text(
# #                 (start_x, current_y), product, fill=(0, 0, 0), font=font
# #             )  # Black color
# #             current_y += line_height + 2  # Move to the next line

# #             # Draw price (slightly indented)
# #             price_x = int(image_width * 0.15)  # Slightly more indented using percentage
# #             draw.text(
# #                 (price_x, current_y),
# #                 f"${details['price']:.2f}",
# #                 fill=(0, 128, 0),
# #                 font=font,
# #             )  # Green color for price
# #             current_y += line_height + 2  # Move to the next line

# #             # Draw description
# #             description = details["description"]
# #             draw.text(
# #                 (start_x, current_y),
# #                 description,
# #                 fill=(50, 50, 50),
# #                 font=font,
# #             )  # Gray color
# #             current_y += line_height + 10  # Add some extra space after the description

# #         img.save(image_file_path)

# #     except Exception as e:
# #         print(f"Error opening or drawing on image: {e}")
# #         return image_file_path  # Return the path even if drawing fails for now
# import os
# from google import genai
# from google.genai import types


# client = genai.Client(api_key=GEMINI_API_KEY)


# model_instructions = """
# Generate a visually appealing menu for Scar's Bakery with interleaving Thought, Action, Observation steps. Thought can reason about the current situation,
# Observation is understanding relevant information from an Action's output and Action can be one of three types:

# (1) database, read and understand the database compenants if it exist search the database for all keys and elements the keys are items and elements are prices.
#     it's impossible to duplicated two keys but it's okay to have have duplicated elements. create a list of items and prices and be aware of the list length.
# (2) style, according to the length from the list you have to create design cartoonish and playful illustration with bright and cheerful colors.
#     make sure "Scar's Bakery" is prominently displayed in a playful and slightly mischievous-looking font. Include a small, cute cartoon lion with a friendly expression.
#     leaving a blank space for the next step.
# (3) menu, now combine the previous two steps to get the menu take the list we got from database and add it into the the blank space from style step.
# """

# example1 = """database
# {"Sourdough Loaf": "$2.00", "Multigrain Bread": "$3.00", "Croissant": "$4.00", "Blueberry Scone": "$4.00"}
# Tell me about Marie Curie and what fields did she win Nobel Prizes in?


# Thought 1
# I need to understand the database compenants

# Action 1
# <search>Marie Curie</search>

# Observation 1
# Marie Curie (/ˈkjʊəri/ KEWR-ee, French: [maʁi kyʁi]; née Maria Skłodowska; 7 November 1867 – 4 July 1934) was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two different scientific fields. Her husband Pierre Curie was a co-winner of her first Nobel Prize. Her daughter Irène Joliot-Curie and son-in-law Frédéric Joliot-Curie also won Nobel Prizes.

# Thought 2
# The first paragraph mentions she won Nobel Prizes in two different scientific fields and that her husband shared the first one. I need to find out the specific fields and the year of the second one. I should look for "Nobel Prize" to get more details.

# Action 2
# <lookup>Nobel Prize</lookup>

# Observation 2
# In 1903, she was awarded the Nobel Prize in Physics, along with her husband Pierre Curie and Henri Becquerel, "in recognition of the extraordinary services they have rendered by their joint research on the radiation phenomena discovered by Professor Henri Becquerel." In 1911, she received the Nobel Prize in Chemistry "in recognition of her services to the advancement of chemistry by the discovery of the elements radium and polonium by the isolation of radium and the study of the nature and compounds of this remarkable element."

# Thought 3
# I have found the two Nobel Prizes and the fields she won them in. The first was in Physics in 1903, and the second was in Chemistry in 1911.

# Action 3
# <finish>Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.</finish>"""


# example2 = """Question
# Tell me about MonkeyDLuffy and what is his dream?

# Thought 1
# I need to search MonkeyDLuffy, find out who he is, then search for his dream.

# Action 1
# <search>MonkeyDLuffy</search>

# Observation 1
# Monkey D. Luffy (/ˈluːfi/ LOO-fee) (Japanese: モンキー・D・ルフィ, Hepburn: Monkī Dī Rufi, [ɾɯꜜɸiː]), also known as "Straw Hat" Luffy,

# Thought 2
# It does not mention his dream. So I need to look up dream.

# Action 2
# <lookup>dream</lookup>

# Observation 2
# Monkey D. Luffy is the captain of the Straw Hat Pirates, and dreamt of being a pirate since childhood from the influence of his idol and mentor Red-Haired Shanks

# Thought 3
# His name is Monkey D. Luffy, also known as "Straw Hat" Luffy. Monkey D. Luffy is the captain of the Straw Hat Pirates, and has dreamt of being a pirate since childhood.

# Action 5
# <finish>Monkey D. Luffy is the captain of the Straw Hat Pirates, and his dream is to be the pirate king</finish>
# """
# question = """Question
# Who is the hero of Konoha and is he married?
# """

# # You will perform the Action; so generate up to, but not including, the Observation.
# react_config = types.GenerateContentConfig(
#     system_instruction=model_instructions + example1 + example2,
# )

# # Create a chat that has the model instructions and examples pre-seeded.
# react_chat = client.chats.create(
#     model="gemini-2.0-flash",
#     config=react_config,
# )

# resp = react_chat.send_message(question)
# print(resp.text)
