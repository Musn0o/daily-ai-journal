from google import genai
import datetime


class Calendar:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.now = datetime.datetime.now()

    def get_today_occasion(self):
        current_month = self.now.month
        current_day = self.now.day

        date_str = f"{current_month}/{current_day}"
        prompt = f"""
        Today's date is {date_str}. Please identify if this date corresponds to any significant international special days or observances.

        If it is a special day, return the name of the occasion followed by the delimiter "|||" and then a short, friendly welcome message for Scar's Bakery related to that occasion. For example: "Mother's Day|||Happy Mother's Day! Treat your amazing mom to something sweet today at Scar's Bakery!"

        If it is not a special day, return "None|||" followed by a generic friendly welcome message for Scar's Bakery, such as "None|||Welcome to Scar's Bakery! Enjoy our delicious treats today."

        Keep the welcome message concise and suitable for a bakery.
        """

        response = (
            self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,  # Use the dynamic prompt here
            ),
        )
        try:
            result = response[0].text.strip().split("|||")
            if len(result) == 2:
                occasion_name = result[0].strip()
                welcome_message = result[1].strip()
                if occasion_name.lower() == "none":
                    if current_month == 6:  # Explicitly check for Pride Month
                        return (
                            "Pride Month",
                            "Celebrate Pride Month with colorful treats from Scar's Bakery!",
                        )
                    return None, welcome_message
                else:
                    return occasion_name, welcome_message
            else:
                return (
                    None,
                    "Welcome to Scar's Bakery! Enjoy our delicious treats today.",
                )
        except Exception as e:
            print(
                f"An error occurred while trying to get today's occasion: {e}"
            )  # Or use logging
            return None, "Welcome to Scar's Bakery! Enjoy our delicious treats today."


# if __name__ == "__main__":
#     # Example usage (you'll need to initialize your Gemini Pro model)
#     # from google.generativeai import configure
#     # configure(api_key="YOUR_API_KEY")
#     # model = GenerativeModel("gemini-pro")
#     # month = 4
#     # day = 13
#     # occasion, message = get_today_occasion(month, day, model)
#     # if occasion:
#     #     print(f"Today is {occasion}! Welcome message: {message}")
#     # else:
#     #     print(message)
#     pass
