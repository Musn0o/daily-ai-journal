# At the beginning of database_manager.py
import os
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from google import genai
from google.api_core import retry
from database_utils import create_connection
# from typing_extensions import TypedDict
# from typing import Annotated, List
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# from langgraph.graph.message import add_messages


DATABASE_NAME = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/data/scar_bakery_ai.db"


is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})  # noqa: E731

if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
        genai.models.Models.generate_content
    )


class Administrator:
    def __init__(self, db_name=DATABASE_NAME):
        self.db_name = db_name
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        print("BakerySupervisor initialized.")

    def list_tables(self):
        """Retrieve the names of all tables in the database."""
        print(" - DB CALL: list_tables()")
        conn = create_connection()
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
        conn = create_connection()
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
        You are an AI bakery administrator. Your task is to manage the database for Scar's Bakery.

        Your first task is to generate the SQL commands to create an SQLite database file named "{self.db_name}" (if it doesn't exist) and the following tables with the specified columns and data types. Ensure that you use the "IF NOT EXISTS" clause when creating each table to avoid errors if the tables already exist:

        1.  Table: menu
            Columns:
                product_name (TEXT, PRIMARY KEY)
                price (REAL)
                description (TEXT)
                quantity (INTEGER)
                availability (TEXT)

        2.  Table: operating_hours
            Columns:
                day_of_week (TEXT, PRIMARY KEY)
                opening_time (TEXT) (format: HH:MM)
                closing_time (TEXT) (format: HH:MM)

        3.  Table: customers
            Columns:
                customer_id (INTEGER PRIMARY KEY AUTOINCREMENT)
                name (TEXT NOT NULL)
                contact_info (TEXT)

        After generating the table creation commands, your next step is to provide instructions on how to populate the 'menu', 'operating_hours', and 'customers' tables. Please provide these instructions as a JSON object. The JSON object should have the table names ('menu', 'operating_hours', 'customers') as keys. For each table, the value should be an object with the following keys:

        - 'description': A brief description of the table.
        - 'columns': An array of objects, where each object describes a column with the following keys:
            - 'name': The name of the column.
            - 'type': The data type of the column.
            - 'description': A description of what data should be entered for this column.
            - 'constraints': (Optional) An array of constraints (e.g., 'PRIMARY KEY', 'NOT NULL').

        Your response should first contain the SQL commands to create the tables, followed by the JSON object containing the data entry instructions. Please separate these sections clearly.
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

        conn = create_connection()
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

    def _prompt_for_menu_data(self):
        """Prompts the user to enter data for the menu table."""
        print("\n--- Menu Table Data Entry ---")
        menu_info = self.table_instructions.get("menu", {})
        print(f"{menu_info.get('description', 'Entering data for the menu table.')}")
        menu_data = []
        entered_product_names = set()
        while True:
            product_name = input(
                "Enter product name (or type 'done' to finish): "
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
            quantity_str = input(f"Enter quantity for '{product_name}': ").strip()
            try:
                quantity = int(quantity_str)
            except ValueError:
                print("Invalid quantity. Please enter a whole number.")
                entered_product_names.remove(product_name)
                continue
            menu_data.append(
                {
                    "product_name": product_name,
                    "price": price,
                    "description": description,
                    "quantity": quantity,
                }
            )
        return menu_data

    def _prompt_for_operating_hours_data(self):
        """Prompts the user to enter data for the operating hours table."""
        print("\n--- Operating Hours Table Data Entry ---")
        operating_hours_info = self.table_instructions.get("operating_hours", {})
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
            return operating_hours_data
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
                        print("Invalid time format. Please use HH:MM (e.g., 08:00).")
                while True:
                    closing_time_sun = input(
                        "Enter closing time for Sunday (HH:MM): "
                    ).strip()
                    if self.validate_time_format(closing_time_sun):
                        break
                    else:
                        print("Invalid time format. Please use HH:MM (e.g., 14:00).")
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
                        print("Invalid time format. Please use HH:MM (e.g., 07:00).")
                while True:
                    closing_time = input(
                        f"Enter closing time for {day} (HH:MM): "
                    ).strip()
                    if self.validate_time_format(closing_time):
                        break
                    else:
                        print("Invalid time format. Please use HH:MM (e.g., 18:00).")
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
                        print("Invalid time format. Please use HH:MM (e.g., 08:00).")
                while True:
                    closing_time_sun = input(
                        "Enter closing time for Sunday (HH:MM): "
                    ).strip()
                    if self.validate_time_format(closing_time_sun):
                        break
                    else:
                        print("Invalid time format. Please use HH:MM (e.g., 14:00).")
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
            print("Invalid choice for operating hours setup.")
            return operating_hours_data

        return operating_hours_data

    def prompt_for_manager_data(self):
        print("\n--- Starting Manager Data Entry ---")

        skip_setup = (
            input("Would you like to set up the initial bakery data now? (yes/no): ")
            .strip()
            .lower()
        )
        if skip_setup == "no":
            print(
                "\nOkay, skipping data setup for now. You can modify the database later through the Document Controller."
            )
            return

        # --- Menu Table ---
        if self.table_instructions and "menu" in self.table_instructions:
            menu_data = self._prompt_for_menu_data()
            if menu_data:
                while True:
                    print("\nHere's the data you entered for the menu:")
                    for item in menu_data:
                        print(item)
                    confirm = (
                        input(
                            "Would you like to confirm and add this data to the menu table? (yes/no): "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm == "yes":
                        self.add_data_to_table("menu", menu_data)
                        break
                    else:
                        re_enter = (
                            input(
                                "Would you like to re-enter the menu items? (yes/no): "
                            )
                            .strip()
                            .lower()
                        )
                        if re_enter == "yes":
                            menu_data = (
                                self._prompt_for_menu_data()
                            )  # Call the function again to re-enter
                        else:
                            print("Setup complete.")
                            break
            else:
                print("No menu data entered.")
        else:
            print("Instructions for the 'menu' table not found.")

        # --- Operating Hours Table ---
        if self.table_instructions and "operating_hours" in self.table_instructions:
            operating_hours_data = self._prompt_for_operating_hours_data()
            if operating_hours_data:
                while True:
                    print("\nHere's the data you entered for the operating hours:")
                    for item in operating_hours_data:
                        print(item)
                    confirm = (
                        input(
                            "Would you like to confirm and add this data to the operating hours table? (yes/no): "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm == "yes":
                        self.add_data_to_table("operating_hours", operating_hours_data)
                        break
                    else:
                        re_enter = (
                            input(
                                "Would you like to re-enter the operating hours? (yes/no): "
                            )
                            .strip()
                            .lower()
                        )
                        if re_enter == "yes":
                            operating_hours_data = (
                                self._prompt_for_operating_hours_data()
                            )
                        else:
                            print("Proceeding to the next step.")
                            break
            else:
                print("No operating hours data entered or setup skipped.")
        else:
            print("Instructions for the 'operating_hours' table not found.")

    def validate_time_format(self, time_str):
        import re

        return re.match(r"^\d{2}:\d{2}$", time_str) is not None

    def add_data_to_table(self, table_name, data):
        """Adds data to the specified table."""
        print(f"\n--- Adding data to {table_name} ---")
        if not data:
            print("No data provided to add.")
            return

        conn = create_connection()
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


#     def start_admin_mode(self):
#         print("Hi from BakerySupervisor's start_admin_mode")
#         self.admin_interface = BakeryAdminInterface(
#             self, self.model
#         )  # Pass the supervisor and the model instance
#         self.admin_interface.start_admin_interaction()


# class AdminState(TypedDict):
#     messages: Annotated[List[BaseMessage], add_messages]


# class BakeryAdminInterface:
#     def __init__(self, supervisor, model):  # Receive the model instance
#         self.supervisor = supervisor
#         self.api_key = supervisor.api_key
#         self.model = model  # Use the passed model instance
#         self.system_prompt = "Hello, admin."  # Keeping it simple for now

#     def initial_greeting_node(self, state: AdminState):
#         response = self.model.invoke([SystemMessage(content=self.system_prompt)])
#         return {"messages": [response]}

#     def start_admin_interaction(self):
#         print("Starting simplified admin interaction...")
#         graph_builder = StateGraph(AdminState)
#         graph_builder.add_node("initial_greeting", self.initial_greeting_node)
#         graph_builder.set_entry_point("initial_greeting")
#         graph_builder.add_edge("initial_greeting", END)
#         self.admin_graph = graph_builder.compile()
#         config = {"recursion_limit": 100}
#         final_state = self.admin_graph.invoke({"messages": []}, config)
#         print("Simplified Admin Interaction Finished.")
#         print("Final State:", final_state)


# class AdminState(TypedDict):
#     messages: Annotated[List[BaseMessage], add_messages]


# class BakeryAdminInterface:
#     def __init__(self, supervisor):
#         self.supervisor = supervisor
#         self.api_key = supervisor.api_key
#         self.model = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash", google_api_key=self.api_key
#         )
#         self.system_prompt = "Hello, admin."  # Simplified prompt
#         self.graph_builder = StateGraph(AdminState)
#         self.admin_graph = None  # We'll compile the graph later

#     def initial_greeting_node(self, state: AdminState):
#         model = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash", google_api_key=self.api_key
#         )
#         response = model.invoke([SystemMessage(content=self.system_prompt)])
#         return {"messages": [response]}

#     def human_input_node_admin(self, state: AdminState):
#         last_msg = state["messages"][-1]
#         if isinstance(last_msg, AIMessage):
#             print("Admin Assistant:", last_msg.content)
#         user_input = input("Owner: ")
#         return {"messages": [HumanMessage(content=user_input)]}

#     def admin_task_router_node(self, state: AdminState):
#         """Routes the admin interaction based on the owner's input."""
#         last_msg = state["messages"][-1]
#         if isinstance(last_msg, HumanMessage):
#             user_input = last_msg.content
#             print(f"Owner's input: {user_input}")
#             # We'll use the model to understand the intent and decide the next step
#             # For now, let's just hardcode a response to test the routing
#             if "menu" in user_input.lower():
#                 return {"next_task": "update_menu"}
#             elif "hours" in user_input.lower():
#                 return {"next_task": "update_hours"}
#             elif "report" in user_input.lower():
#                 return {"next_task": "view_reports"}
#             elif "exit" in user_input.lower():
#                 return {"next_task": "exit_admin"}
#             else:
#                 return {"next_task": "unknown_command"}
#         else:
#             # This should ideally not happen here, but let's handle it just in case
#             return {"next_task": "unknown"}

#     def start_admin_interaction(self):
#         print("Starting admin interaction...")

#         self.graph_builder.add_node("initial_greeting", self.initial_greeting_node)
#         self.graph_builder.add_node("owner_input", self.human_input_node_admin)
#         self.graph_builder.add_node("task_router", self.admin_task_router_node)

#         self.graph_builder.add_edge(START, "initial_greeting")
#         self.graph_builder.add_edge("initial_greeting", "owner_input")
#         self.graph_builder.add_edge("owner_input", "task_router")

#         # Add conditional edges from the task router
#         self.graph_builder.add_conditional_edges(
#             "task_router",
#             lambda state: state.get("next_task"),
#             {
#                 "update_menu": "handle_update_menu",  # We'll create this node later
#                 "update_hours": "handle_update_hours",  # We'll create this node later
#                 "view_reports": "handle_view_reports",  # We'll create this node later
#                 "exit_admin": END,  # For now, exiting will end the admin session
#                 "unknown_command": "owner_input",  # If the command is not recognized, go back for more input
#                 "unknown": "owner_input",  # Handle the 'unknown' case as well
#             },
#         )

#         # For now, let's add a simple placeholder node for update_menu
#         def handle_update_menu_node(state):
#             print("Handling update menu...")
#             return {}

#         self.graph_builder.add_node("handle_update_menu", handle_update_menu_node)
#         self.graph_builder.add_edge(
#             "handle_update_menu", "owner_input"
#         )  # Go back for more

#         # Placeholder for update_hours
#         def handle_update_hours_node(state):
#             print("Handling update hours...")
#             return {}

#         self.graph_builder.add_node("handle_update_hours", handle_update_hours_node)
#         self.graph_builder.add_edge(
#             "handle_update_hours", "owner_input"
#         )  # Go back for more

#         # Placeholder for view_reports
#         def handle_view_reports_node(state):
#             print("Handling view reports...")
#             return {}

#         self.graph_builder.add_node("handle_view_reports", handle_view_reports_node)
#         self.graph_builder.add_edge(
#             "handle_view_reports", "owner_input"
#         )  # Go back for more

#         self.admin_graph = self.graph_builder.compile()
#         config = {"recursion_limit": 100}
#         final_state = self.admin_graph.invoke({"messages": []}, config)
#         print("Final Admin State:", final_state)


if __name__ == "__main__":
    supervisor = Administrator(DATABASE_NAME)
    supervisor.run()
