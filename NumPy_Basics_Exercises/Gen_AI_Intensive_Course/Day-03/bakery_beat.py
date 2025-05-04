import os
from google import genai
import datetime
import pytz
from database_utils import create_connection
from calendar_helper import Calendar


class BakeryBeat:
    def __init__(self):
        GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # self.menu_designer = Designer()
        self.organizer = Calendar(GEMINI_API_KEY)
        self.now = datetime.datetime.now()

    def is_bakery_open(self):
        now = datetime.datetime.now(pytz.timezone("EET"))  # Get current time in EET
        # Note for users running this code in a different time zone:
        # The bakery's operating hours are currently being checked against the Eastern European Time (EET) zone.
        # If the bakery operates in a different time zone, you may need to:
        # 1. Modify the timezone in the line above (e.g., to your local timezone using pytz.timezone("Your_Timezone")).
        # 2. Ensure the operating hours stored in the database are also in that timezone.
        # 3. Alternatively, consider using UTC for timekeeping and converting to the bakery's local timezone.
        current_day = now.strftime("%A")  # Get full day name (e.g., Monday)
        current_time_str = now.strftime("%H:%M")  # Get current time in HH:MM format
        current_time_obj = datetime.datetime.strptime(current_time_str, "%H:%M").time()

        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT opening_time, closing_time FROM operating_hours WHERE day_of_week=?",
                (current_day,),
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                opening_time_str, closing_time_str = result
                if opening_time_str is None or closing_time_str is None:
                    return False  # Assume closed if no hours are set or if set to None

                try:
                    opening_time = datetime.datetime.strptime(
                        opening_time_str, "%H:%M"
                    ).time()
                    closing_time = datetime.datetime.strptime(
                        closing_time_str, "%H:%M"
                    ).time()
                    print(opening_time, current_time_obj, closing_time)

                    if closing_time < opening_time:
                        # Closing time is on the next day
                        if (
                            opening_time <= current_time_obj
                            or current_time_obj <= closing_time
                        ):
                            return True
                        else:
                            return False
                    else:
                        # Closing time is on the same day
                        if opening_time <= current_time_obj <= closing_time:
                            return True
                        else:
                            return False

                except ValueError as e:
                    print(
                        f"Error parsing time for {current_day}: Opening - {opening_time_str}, Closing - {closing_time_str}. Error: {e}"
                    )
                    return False
            else:
                return False  # Assume closed if no entry for the current day

        return False  # Return False if connection fails

    # def generate_welcome_message(self):
    #     current_month = self.now.month

    #     occasion_name, welcome_message = self.organizer.get_today_occasion()

    #     image_path = self.menu_designer.generate_welcome_image(
    #         month=current_month, occasion_name=occasion_name
    #     )

    #     if image_path:
    #         WELCOME_MSG = f"{welcome_message}\n\n{image_path}\n\nType `Bye` to quit. How may I serve you today?"
    #         return WELCOME_MSG
    #     else:
    #         WELCOME_MSG = (
    #             f"{welcome_message} Type `Bye` to quit. How may I serve you today?"
    #         )
    #         return WELCOME_MSG
    def generate_welcome_message(self):
        # current_month is no longer directly used here, but keeping self.now might be useful
        # current_month = self.now.month # This line can be removed

        # Get occasion name and welcome message text from Calendar
        occasion_name, welcome_message_text = self.organizer.get_today_occasion()

        # Remove the call to Designer
        # image = self.menu_designer.generate_welcome_image(
        #     month=current_month, occasion_name=occasion_name
        # )

        # Remove the image handling and return only the formatted text message
        WELCOME_MSG = (
            f"{welcome_message_text}\n\nType `Bye` to quit. How may I serve you today?"
        )
        return WELCOME_MSG
