from google import genai
from PIL import Image

""" Text generation """

"""1.Text input"""


client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Do you know how I'm connected with you now?",
)

print(response.text)

"""2.Image input"""

client = genai.Client(api_key="GEMINI_API_KEY")

image = Image.open("Untitled.jpeg")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[image, "Tell me about this instrument"]
)
print(response.text)
