import os
import sqlite3
from typing import Optional
from database_utils import create_connection, get_menu_items, add_customer
from prompts import document_controller_prompt
from google import genai
from google.genai import types


class DocumentController:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        self.tools = [
            self.add_menu_item,
            self.remove_menu_item,
            self.update_menu_item,
            self.update_product_quantity_after_order,
            self.update_operating_hours,
            add_customer,
            self.update_customer,
            get_menu_items,
            self.get_operating_hours,
            self.get_customers,
            self.get_customer_count,
            self.get_bakery_report,
        ]
        self.chat = self.start_chat()

    def start_chat(self):
        return self.client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=document_controller_prompt,
                tools=self.tools,
            ),
        )

    def add_menu_item(
        self, product_name: str, price: float, description: str, quantity: int
    ):
        """Adds a new item to the menu table."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO menu (product_name, price, description, quantity) VALUES (?, ?, ?, ?)",
                    (product_name, price, description, quantity),
                )
                availability = "Available" if quantity > 0 else "Out of Stock"
                cursor.execute(
                    "UPDATE menu SET availability = ? WHERE product_name = ?",
                    (availability, product_name),
                )
                conn.commit()
                conn.close()
                return f"Successfully added '{product_name}' to the menu."
            except sqlite3.IntegrityError as e:
                conn.rollback()
                conn.close()
                return f"Error: Product '{product_name}' already exists in the menu."
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error adding '{product_name}': {e}"
        else:
            return "Error: Could not connect to the database."

    def remove_menu_item(self, product_name: str):
        """Removes an item from the menu table."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "DELETE FROM menu WHERE product_name = ?", (product_name,)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return f"Successfully removed '{product_name}' from the menu."
                else:
                    conn.close()
                    return f"Error: Product '{product_name}' not found in the menu."
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error removing '{product_name}': {e}"
        else:
            return "Error: Could not connect to the database."

    def update_menu_item(
        self,
        product_name: str,
        price: Optional[float] = None,
        description: Optional[str] = None,
        quantity: Optional[int] = None,
    ):
        """Updates the details of a menu item."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            update_fields = []
            update_values = []

            if price is not None:
                update_fields.append("price = ?")
                update_values.append(price)
            if description is not None:
                update_fields.append("description = ?")
                update_values.append(description)
            if quantity is not None:
                update_fields.append("quantity = ?")
                update_values.append(quantity)

            if not update_fields:
                conn.close()
                return "No fields provided to update."

            sql = f"UPDATE menu SET {', '.join(update_fields)} WHERE product_name = ?"
            update_values.append(product_name)

            try:
                cursor.execute(sql, tuple(update_values))
                if cursor.rowcount > 0:
                    # Update availability if quantity was changed
                    if quantity is not None:
                        availability = "Available" if quantity > 0 else "Out of Stock"
                        cursor.execute(
                            "UPDATE menu SET availability = ? WHERE product_name = ?",
                            (availability, product_name),
                        )
                    conn.commit()
                    conn.close()
                    return f"Successfully updated '{product_name}'."
                else:
                    conn.close()
                    return f"Error: Product '{product_name}' not found in the menu."
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error updating '{product_name}': {e}"
        else:
            return "Error: Could not connect to the database."

    def update_product_quantity_after_order(
        self, product_name: str, quantity_change: int
    ):
        """Updates the quantity of a product after an order and adjusts availability."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE menu SET quantity = quantity + ? WHERE product_name = ?",
                    (quantity_change, product_name),
                )
                if cursor.rowcount > 0:
                    # Update availability based on the new quantity
                    cursor.execute(
                        "SELECT quantity FROM menu WHERE product_name = ?",
                        (product_name,),
                    )
                    result = cursor.fetchone()
                    if result:
                        new_quantity = result[0]
                        availability = (
                            "Available" if new_quantity > 0 else "Out of Stock"
                        )
                        cursor.execute(
                            "UPDATE menu SET availability = ? WHERE product_name = ?",
                            (availability, product_name),
                        )
                        conn.commit()
                        conn.close()
                        return f"Successfully updated quantity for '{product_name}' by {quantity_change}."
                    else:
                        conn.close()
                        return f"Error: Could not retrieve updated quantity for '{product_name}'."
                else:
                    conn.close()
                    return f"Error: Product '{product_name}' not found in the menu."
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error updating quantity for '{product_name}': {e}"
        else:
            return "Error: Could not connect to the database."

    def update_operating_hours(
        self,
        day_of_week: str,
        opening_time: Optional[str] = None,
        closing_time: Optional[str] = None,
    ):
        """Updates the opening or closing time for a specific day of the week."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            update_fields = []
            update_values = []

            if opening_time is not None:
                update_fields.append("opening_time = ?")
                update_values.append(opening_time)
            elif opening_time is None:
                update_fields.append("opening_time = ?")
                update_values.append(None)  # Explicitly set to None in the query

            if closing_time is not None:
                update_fields.append("closing_time = ?")
                update_values.append(closing_time)
            elif closing_time is None:
                update_fields.append("closing_time = ?")
                update_values.append(None)  # Explicitly set to None in the query

            if not update_fields:
                conn.close()
                return f"No fields provided to update for {day_of_week}."

            sql = f"UPDATE operating_hours SET {', '.join(update_fields)} WHERE day_of_week = ?"
            update_values.append(day_of_week)

            try:
                cursor.execute(sql, tuple(update_values))
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return f"Successfully updated operating hours for {day_of_week}."
                else:
                    conn.close()
                    return (
                        f"Error: Day '{day_of_week}' not found in the operating hours."
                    )
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error updating operating hours for {day_of_week}: {e}"
        else:
            return "Error: Could not connect to the database."

    def get_operating_hours(self):
        """Retrieves the current operating hours for each day of the week."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT day_of_week, opening_time, closing_time FROM operating_hours"
                )
                results = cursor.fetchall()
                conn.close()
                if results:
                    hours = []
                    for row in results:
                        hours.append(f"- {row[0]}: {row[1]} - {row[2]}")
                    return "Our current operating hours are:\n" + "\n".join(hours)
                else:
                    return "Operating hours are not currently set."
            except sqlite3.Error as e:
                conn.close()
                return f"Database error retrieving operating hours: {e}"
        else:
            return "Error: Could not connect to the database."

    def update_customer(self, name: str, new_contact_info: str):
        """Updates the contact information for an existing customer."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE customers SET contact_info = ? WHERE name = ?",
                    (new_contact_info, name),
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return f"Successfully updated contact info for customer '{name}'."
                else:
                    conn.close()
                    return f"Error: Customer '{name}' not found."
            except sqlite3.Error as e:
                conn.rollback()
                conn.close()
                return f"Database error updating customer '{name}': {e}"
        else:
            return "Error: Could not connect to the database."

    def get_customers(self):
        """Retrieves a list of all customers and their contact information."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT name, contact_info FROM customers")
                results = cursor.fetchall()
                conn.close()
                if results:
                    customers = []
                    for row in results:
                        customers.append(f"- {row[0]}: {row[1]}")
                    return "Our current customers are:\n" + "\n".join(customers)
                else:
                    return "There are currently no customers in our list."
            except sqlite3.Error as e:
                conn.close()
                return f"Database error retrieving customer list: {e}"
        else:
            return "Error: Could not connect to the database."

    def get_customer_count(self):
        """Retrieves the total number of customers."""
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM customers")
                result = cursor.fetchone()
                conn.close()
                if result:
                    return f"Total number of customers: {result[0]}"
                else:
                    return "No customers found."
            except sqlite3.Error as e:
                conn.close()
                return f"Database error retrieving customer count: {e}"
        else:
            return "Error: Could not connect to the database."

    def get_bakery_report(self):
        """Generates a complete report of the bakery's current status in Markdown format."""
        menu_report = get_menu_items()
        hours_report = self.get_operating_hours()
        customers_report = self.get_customer_count()

        report = "# Scar's Bakery - Complete Status Report\n\n"
        report += "## Menu Items:\n"
        report += (
            menu_report.replace("Current menu items:\n", "") + "\n\n"
        )  # Clean up the initial message
        report += "## Operating Hours:\n"
        report += (
            hours_report.replace("Our current operating hours are:\n", "") + "\n\n"
        )  # Clean up the initial message
        report += "## Customer Information:\n"
        report += customers_report + "\n"

        return report


if __name__ == "__main__":
    admin_tool = DocumentController()
    print("Welcome to the Scar's Bakery Document Controller! 🍰")
    print("Type 'exit' to end the session.")

    while True:
        user_input = input("Admin: ")
        if user_input.lower() == "exit":
            break

        response = admin_tool.chat.send_message(user_input)

        if response.text:
            print(f"Gem: {response.text}")
        elif response.candidates[0].content.parts[0].function_call:
            call = response.candidates[0].content.parts[0].function_call
            tool_name = call.name
            arguments = call.args

            print(f"Gem is calling tool: {tool_name} with arguments: {arguments}")

            tool_result = getattr(admin_tool, tool_name)(
                **arguments
            )  # Dynamically call the method

            if tool_result:
                print(f"Tool Result: {tool_result}")
                response_after_tool = admin_tool.chat.send_message(
                    tool_result,
                    generation_config=types.GenerationConfig(
                        response_style=types.ResponseStyle.UPDATE_PROMPT,
                    ),
                )
                if response_after_tool.text:
                    print(f"Gem: {response_after_tool.text}")
            else:
                print("Gem: Hmm, that tool call didn't seem to work correctly.")
        else:
            print("Gem: Sorry, I'm not sure how to respond to that.")

    print("Exiting Scar's Bakery Document Controller. 👋")
