import os
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import datetime
import pytz  # You might need to install this: pip install pytz
from google import genai
from google.api_core import retry

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
DATABASE_NAME = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/data/scar_bakery_ai.db"
SECRET_KEY = "Scar's Bakery Rules"  # Let's define the secret key here for now

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})  # noqa: E731

if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )


class BakerySupervisor:
    def __init__(
        self, db_name=DATABASE_NAME, api_key=GEMINI_API_KEY, secret_key=SECRET_KEY
    ):
        self.db_name = db_name
        self.api_key = api_key
        self.secret_key = secret_key
        print("BakerySupervisor initialized.")

    def create_connection(self):
        """Creates a connection to the SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            print(f"Connected to database: {self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
        return conn

    def list_tables(self):
        """Retrieve the names of all tables in the database."""
        print(" - DB CALL: list_tables()")
        conn = self.create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            return [t[0] for t in tables]
        else:
            return []

    def ask_ai_for_schema(self, prompt):
        """Sends the prompt to the Gemini model and returns the response."""
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", google_api_key=self.api_key
        )
        response = model.invoke([("user", prompt)])
        return response.content

    def execute_sql_commands(self, conn, sql_commands):
        """Executes a list of SQL commands."""
        cursor = conn.cursor()
        try:
            for command in sql_commands:
                cursor.execute(command)
            conn.commit()
            print("SQL commands executed successfully.")
        except sqlite3.Error as e:
            print(f"Error executing SQL: {e}")

    def describe_table(self, table_name):
        """Look up the table schema."""
        print(f" - DB CALL: describe_table({table_name})")
        conn = self.create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()
            conn.close()
            return [(col[1], col[2]) for col in schema]
        else:
            return []

    def create_database_and_tables(self):
        initial_prompt = """
            You are an AI bakery supervisor. Your task is to manage the database for Scar's Bakery.

            Your first task is to generate the SQL commands to create an SQLite database file named "{self.db_name}" (if it doesn't exist) and the following tables with the specified columns and data types. Ensure that you use the "IF NOT EXISTS" clause when creating each table to avoid errors if the tables already exist:

            1.  Table: menu
                Columns:
                    product_name (TEXT, PRIMARY KEY)
                    price (REAL)
                    description (TEXT)

            2.  Table: operating_hours
                Columns:
                    day_of_week (TEXT, PRIMARY KEY)
                    opening_time (TEXT) (format: HH:MM)
                    closing_time (TEXT) (format: HH:MM)

            3.  Table: product_status
                Columns:
                    product_name (TEXT, PRIMARY KEY, FOREIGN KEY references menu(product_name))
                    status (TEXT) (e.g., "Available", "Out of Stock", "Seasonal")
                    availability_description (TEXT)

            4.  Table: customers
                Columns:
                    customer_id (INTEGER PRIMARY KEY AUTOINCREMENT)
                    name (TEXT NOT NULL)
                    contact_info (TEXT)

            5.  Table: staff
                Columns:
                    staff_id (INTEGER PRIMARY KEY AUTOINCREMENT)
                    first_name (TEXT NOT NULL)
                    last_name (TEXT NOT NULL)
                    role (TEXT)

            After generating the table creation commands, your next step is to provide instructions on how to populate the 'menu', 'operating_hours', and 'staff' tables. Please provide these instructions as a JSON object. The JSON object should have the table names ('menu', 'operating_hours', 'staff') as keys. For each table, the value should be an object with the following keys:

            - 'description': A brief description of the table.
            - 'columns': An array of objects, where each object describes a column with the following keys:
                - 'name': The name of the column.
                - 'type': The data type of the column.
                - 'description': A description of what data should be entered for this column.
                - 'constraints': (Optional) An array of constraints (e.g., 'PRIMARY KEY', 'NOT NULL').

            Furthermore, any subsequent request to modify the data in these tables (inserting, updating, or deleting records) will only be executed if I, the bakery owner, provide the secret key: "{self.secret_key}". Please make sure to mention this requirement in your initial response after the JSON instructions.

            Your response should first contain the SQL commands to create the tables, followed by the JSON object containing the data entry instructions, and finally, the instruction about the secret key for future modifications. Please separate these sections clearly.
        """.format(self=self)

        ai_response = self.ask_ai_for_schema(initial_prompt)
        print("AI Response:\n", ai_response)

        # Extract SQL commands (same as before)
        sql_commands_str = ""
        sql_start = ai_response.find("```sql")
        sql_end = ai_response.find("```", sql_start + 1)
        if sql_start != -1 and sql_end != -1:
            sql_commands_str = ai_response[sql_start + len("```sql") : sql_end].strip()
            sql_commands = sql_commands_str.split(";")
            sql_commands = [cmd.strip() + ";" for cmd in sql_commands if cmd.strip()]
        else:
            sql_commands = []

        conn = self.create_connection()
        if conn:
            self.execute_sql_commands(conn, sql_commands)
            print("\n--- Tables in the database ---")
            tables = self.list_tables()
            for table in tables:
                print(f"- {table}")
            conn.close()

        # Extract JSON instructions
        json_start = ai_response.find("{")
        json_end = ai_response.rfind("}")  # Find the last occurrence of '}'
        self.table_instructions = {}
        if json_start != -1 and json_end != -1 and json_start < json_end:
            json_str = ai_response[json_start : json_end + 1]
            try:
                self.table_instructions = json.loads(json_str)
                print("\n--- Parsed JSON Instructions ---")
                print(json.dumps(self.table_instructions, indent=4))
            except json.JSONDecodeError as e:
                print(f"\n--- Error decoding JSON instructions: {e} ---")
                print("Raw JSON String:\n", json_str)
        else:
            print("\n--- JSON instructions not found in the AI response. ---")

        print(
            "\n--- Starting Manager Data Entry (after JSON parsing) ---"
        )  # For testing

    def prompt_for_manager_data(self):
        print("\n--- Starting Manager Data Entry ---")

        # --- Menu Table ---
        if self.table_instructions and "menu" in self.table_instructions:
            menu_info = self.table_instructions["menu"]
            print(f"\n--- Menu Table Data Entry ---")
            print(
                f"{menu_info.get('description', 'Entering data for the menu table.')}"
            )
            menu_data = []
            entered_product_names = set()
            while True:
                product_name = input(
                    f"Enter product name (or type 'done' to finish): "
                ).strip()
                if product_name.lower() == "done":
                    break
                if product_name in entered_product_names:
                    print(
                        f"Error: The product name '{product_name}' has already been entered in this session. Please enter a different name."
                    )
                    continue
                if not product_name:
                    print("Product name cannot be empty.")
                    continue
                entered_product_names.add(product_name)
                price_str = input(f"Enter price for '{product_name}': ").strip()
                try:
                    price = float(price_str)
                except ValueError:
                    print("Invalid price. Please enter a number.")
                    entered_product_names.remove(product_name)
                    continue
                description = input(f"Enter description for '{product_name}': ").strip()
                menu_data.append(
                    {
                        "product_name": product_name,
                        "price": price,
                        "description": description,
                    }
                )
            if menu_data:
                self.add_data_to_table("menu", menu_data)
        else:
            print("Instructions for the 'menu' table not found.")

        # --- Operating Hours Table ---
        if self.table_instructions and "operating_hours" in self.table_instructions:
            operating_hours_info = self.table_instructions["operating_hours"]
            print(f"\n--- Operating Hours Table Data Entry ---")
            print(
                f"{operating_hours_info.get('description', 'Entering data for the operating hours table.')}"
            )
            operating_hours_data = []
            days_of_week = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]

            setup_choice = (
                input(
                    "Would you like to setup time yourself or you need a quick setup? (self/quick/done to skip): "
                )
                .strip()
                .lower()
            )

            if setup_choice == "done":
                print("Skipping operating hours setup.")
            elif setup_choice == "quick":
                print("\n--- Quick Setup for Weekdays (Monday to Saturday) ---")
                while True:
                    opening_time_quick = input(
                        "Enter opening time for weekdays (HH:MM): "
                    ).strip()
                    if self.validate_time_format(opening_time_quick):
                        break
                    else:
                        print("Invalid time format. Please use HH:MM (e.g., 07:00).")
                while True:
                    closing_time_quick = input(
                        "Enter closing time for weekdays (HH:MM): "
                    ).strip()
                    if self.validate_time_format(closing_time_quick):
                        break
                    else:
                        print("Invalid time format. Please use HH:MM (e.g., 18:00).")

                for day in days_of_week:
                    operating_hours_data.append(
                        {
                            "day_of_week": day,
                            "opening_time": opening_time_quick,
                            "closing_time": closing_time_quick,
                        }
                    )

                # Handle Sunday
                print("\n--- Setting hours for Sunday ---")
                sunday_closed = (
                    input("Is the bakery open on Sunday? (yes/no): ").strip().lower()
                )
                if sunday_closed == "yes":
                    while True:
                        opening_time_sun = input(
                            "Enter opening time for Sunday (HH:MM): "
                        ).strip()
                        if self.validate_time_format(opening_time_sun):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 08:00)."
                            )
                    while True:
                        closing_time_sun = input(
                            "Enter closing time for Sunday (HH:MM): "
                        ).strip()
                        if self.validate_time_format(closing_time_sun):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 14:00)."
                            )
                    operating_hours_data.append(
                        {
                            "day_of_week": "Sunday",
                            "opening_time": opening_time_sun,
                            "closing_time": closing_time_sun,
                        }
                    )
                else:
                    print("Sunday will be set as closed.")
                    operating_hours_data.append(
                        {
                            "day_of_week": "Sunday",
                            "opening_time": None,
                            "closing_time": None,
                        }
                    )

            elif setup_choice == "self":
                for day in days_of_week:
                    print(f"\n--- Setting hours for {day} ---")
                    while True:
                        opening_time = input(
                            f"Enter opening time for {day} (HH:MM): "
                        ).strip()
                        if self.validate_time_format(opening_time):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 07:00)."
                            )
                    while True:
                        closing_time = input(
                            f"Enter closing time for {day} (HH:MM): "
                        ).strip()
                        if self.validate_time_format(closing_time):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 18:00)."
                            )
                    operating_hours_data.append(
                        {
                            "day_of_week": day,
                            "opening_time": opening_time,
                            "closing_time": closing_time,
                        }
                    )

                # Handle Sunday
                print("\n--- Setting hours for Sunday ---")
                sunday_closed = (
                    input("Is the bakery open on Sunday? (yes/no): ").strip().lower()
                )
                if sunday_closed == "yes":
                    while True:
                        opening_time_sun = input(
                            "Enter opening time for Sunday (HH:MM): "
                        ).strip()
                        if self.validate_time_format(opening_time_sun):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 08:00)."
                            )
                    while True:
                        closing_time_sun = input(
                            "Enter closing time for Sunday (HH:MM): "
                        ).strip()
                        if self.validate_time_format(closing_time_sun):
                            break
                        else:
                            print(
                                "Invalid time format. Please use HH:MM (e.g., 14:00)."
                            )
                    operating_hours_data.append(
                        {
                            "day_of_week": "Sunday",
                            "opening_time": opening_time_sun,
                            "closing_time": closing_time_sun,
                        }
                    )
                else:
                    print("Sunday will be set as closed.")
                    operating_hours_data.append(
                        {
                            "day_of_week": "Sunday",
                            "opening_time": None,
                            "closing_time": None,
                        }
                    )

            else:
                print("Invalid choice. Skipping operating hours setup.")

            if operating_hours_data:
                self.add_data_to_table("operating_hours", operating_hours_data)
        else:
            print("Instructions for the 'operating_hours' table not found.")

        # --- Staff Table ---
        if self.table_instructions and "staff" in self.table_instructions:
            staff_info = self.table_instructions["staff"]
            print(f"\n--- Staff Table Data Entry ---")
            print(
                f"{staff_info.get('description', 'Entering data for the staff table.')}"
            )
            staff_data = []
            while True:
                first_name = input(
                    "Enter staff first name (or type 'done' to finish): "
                ).strip()
                if first_name.lower() == "done":
                    break
                if not first_name:
                    print("First name cannot be empty.")
                    continue
                last_name = input(f"Enter staff last name for {first_name}: ").strip()
                role = input(f"Enter role for {first_name} {last_name}: ").strip()
                staff_data.append(
                    {"first_name": first_name, "last_name": last_name, "role": role}
                )
            if staff_data:
                self.add_data_to_table("staff", staff_data)
        else:
            print("Instructions for the 'staff' table not found.")

    def validate_time_format(self, time_str):
        import re

        return re.match(r"^\d{2}:\d{2}$", time_str) is not None

    def add_data_to_table(self, table_name, data):
        """Adds data to the specified table."""
        print(f"\n--- Adding data to {table_name} ---")
        if not data:
            print("No data provided to add.")
            return

        conn = self.create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                first_row = data[0]
                columns = ", ".join(first_row.keys())
                placeholders = ", ".join(["?"] * len(first_row))
                sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

                for row in data:
                    values = tuple(row.values())
                    cursor.execute(sql, values)

                conn.commit()
                print(
                    f"Successfully added {len(data)} rows to the '{table_name}' table."
                )
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    print(
                        "Error: The product name you entered already exists in the database. Product names must be unique."
                    )
                else:
                    print(f"Integrity Error adding data to '{table_name}': {e}")
            except sqlite3.Error as e:
                print(f"Error adding data to '{table_name}': {e}")
            finally:
                conn.close()

    def run(self):
        self.create_database_and_tables()
        if self.table_instructions:
            self.prompt_for_manager_data()
        else:
            print(
                "Failed to retrieve data entry instructions from the AI. Data entry will not start."
            )

    def is_bakery_open(self):
        now = datetime.datetime.now(pytz.timezone("EET"))  # Get current time in EET
        current_day = now.strftime("%A")  # Get full day name (e.g., Monday)
        current_time = now.strftime("%H:%M")  # Get current time in HH:MM format

        conn = self.create_connection()
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
                    current_time_obj = datetime.datetime.strptime(
                        current_time, "%H:%M"
                    ).time()

                    if opening_time <= current_time_obj <= closing_time:
                        return True
                    else:
                        return False
                except ValueError:
                    print(
                        f"Error parsing time for {current_day}: Opening - {opening_time_str}, Closing - {closing_time_str}"
                    )
                    return False
            else:
                return False  # Assume closed if no entry for the current day

        return False  # Return False if connection fails

    def start_admin_mode(self):
        print("Hi")


if __name__ == "__main__":
    supervisor = BakerySupervisor(DATABASE_NAME, GEMINI_API_KEY, SECRET_KEY)
    supervisor.run()
