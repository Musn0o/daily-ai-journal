import sqlite3

# Define the database file name
DATABASE_NAME = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/data/scar_bakery_ai.db"


def create_connection():
    """Creates a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Connected to database: {DATABASE_NAME}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn


def get_menu_items():
    """Retrieves all items currently in the menu."""
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT product_name, price, description, quantity, availability FROM menu"
            )
            results = cursor.fetchall()
            conn.close()
            if results:
                menu_items = []
                for row in results:
                    menu_items.append(
                        {
                            "product_name": row[0],
                            "price": row[1],
                            "description": row[2],
                            "quantity": row[3],
                            "availability": row[4],
                        }
                    )
                return "Current menu items:\n" + "\n".join(
                    [
                        f"- {item['product_name']}: ${item['price']:.2f}, Quantity: {item['quantity']} ({item['availability']})"
                        for item in menu_items
                    ]
                )
            else:
                return "The menu is currently empty."
        except sqlite3.Error as e:
            conn.close()
            return f"Database error retrieving menu items: {e}"
    else:
        return "Error: Could not connect to the database."


def process_bakery_order(order_items):
    """Processes a bakery order: updates stock in the menu table (case-insensitive) using the provided connection."""
    try:
        conn = create_connection()
        cursor = conn.cursor()

        for item_name in order_items:
            item_name_lower = item_name.lower()
            print(f"Attempting to order: {item_name} (lowercase: {item_name_lower})")

            cursor.execute(
                "SELECT quantity FROM menu WHERE LOWER(product_name) = ?",
                (item_name_lower,),
            )
            result = cursor.fetchone()

            if result:
                current_quantity = result[0]
                if current_quantity > 0:
                    new_quantity = current_quantity - 1
                    cursor.execute(
                        "UPDATE menu SET quantity = ? WHERE LOWER(product_name) = ?",
                        (new_quantity, item_name_lower),
                    )
                else:
                    print(f"Warning: Out of stock for {item_name}")
            else:
                print(f"Warning: Product '{item_name}' not found in the menu")

        conn.commit()
        return True

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def add_customer(name: str, contact_info: str):
    """Adds a new customer to the customers table."""
    print(f"Adding customer: {name} email: {contact_info}")
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO customers (name, contact_info) VALUES (?, ?)",
                (name, contact_info),
            )
            conn.commit()
            conn.close()
            return f"Successfully added customer '{name}' with contact info '{contact_info}'."
        except sqlite3.Error as e:
            conn.rollback()
            conn.close()
            return f"Database error adding customer '{name}': {e}"
    else:
        return "Error: Could not connect to the database."


# def process_bakery_order(customer_name, customer_email, order_items):
#     """Processes a bakery order: updates stock in the menu table."""
#     try:
#         conn = create_connection()
#         cursor = conn.cursor()

#         for item_name in order_items:
#             cursor.execute(
#                 "SELECT quantity FROM menu WHERE product_name = ?", (item_name,)
#             )
#             result = cursor.fetchone()

#             if result:
#                 current_quantity = result[0]
#                 if current_quantity > 0:
#                     new_quantity = current_quantity - 1
#                     cursor.execute(
#                         "UPDATE menu SET quantity = ? WHERE product_name = ?",
#                         (new_quantity, item_name),
#                     )
#                 else:
#                     print(
#                         f"Warning: Out of stock for {item_name}"
#                     )  # Consider logging this or handling it differently
#             else:
#                 print(
#                     f"Warning: Product '{item_name}' not found in the menu"
#                 )  # Consider logging this

#         conn.commit()
#         conn.close()
#         print(f"Stock updated for order by {customer_name}.")
#         return True

#     except sqlite3.Error as e:
#         print(f"Database error: {e}")
#         if conn:
#             conn.rollback()
#             conn.close()
#         return False
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False


def update_product_quantity_after_order(product_name: str, quantity_change: int):
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
                    availability = "Available" if new_quantity > 0 else "Out of Stock"
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


# def fetch_all_menu_items(conn):
#     """Fetches all menu items from the menu table."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM menu")
#         rows = cursor.fetchall()
#         return rows
#     except sqlite3.Error as e:
#         print(f"Error fetching menu items: {e}")
#         return None
# def create_menu_table(conn):
#     """Creates the menu table in the database if it doesn't exist."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS menu (
#                 product_name TEXT PRIMARY KEY,
#                 price REAL,
#                 description TEXT
#             )
#         """)
#         conn.commit()
#         print("Menu table created successfully")
#     except sqlite3.Error as e:
#         print(f"Error creating menu table: {e}")


# def create_operating_hours_table(conn):
#     """Creates the operating_hours table if it doesn't exist."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS operating_hours (
#                 day_of_week TEXT PRIMARY KEY,
#                 opening_time TEXT,
#                 closing_time TEXT
#             )
#         """)
#         conn.commit()
#         print("Operating hours table created successfully")
#     except sqlite3.Error as e:
#         print(f"Error creating operating hours table: {e}")


# def create_product_status_table(conn):
#     """Creates the product_status table if it doesn't exist."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS product_status (
#                 product_name TEXT PRIMARY KEY,
#                 status TEXT,
#                 availability_description TEXT,
#                 FOREIGN KEY (product_name) REFERENCES menu(product_name)
#             )
#         """)
#         conn.commit()
#         print("Product status table created successfully")
#     except sqlite3.Error as e:
#         print(f"Error creating product status table: {e}")


# def add_menu_item(conn, product_name, price, description):
#     """Adds a new menu item to the menu table."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute(
#             """
#             INSERT INTO menu (product_name, price, description)
#             VALUES (?, ?, ?)
#         """,
#             (product_name, price, description),
#         )
#         conn.commit()
#         print(f"Added item: {product_name} to the menu")
#     except sqlite3.Error as e:
#         print(f"Error adding menu item: {e}")


# def add_operating_hours(conn, day_of_week, opening_time, closing_time):
#     """Adds operating hours for a specific day."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute(
#             """
#             INSERT INTO operating_hours (day_of_week, opening_time, closing_time)
#             VALUES (?, ?, ?)
#         """,
#             (day_of_week, opening_time, closing_time),
#         )
#         conn.commit()
#         print(f"Added operating hours for {day_of_week}")
#     except sqlite3.Error as e:
#         print(f"Error adding operating hours for {day_of_week}: {e}")


# def fetch_operating_hours_by_day(conn, day_of_week):
#     """Fetches the operating hours for a specific day of the week."""
#     try:
#         cursor = conn.cursor()
#         cursor.execute(
#             "SELECT opening_time, closing_time FROM operating_hours WHERE day_of_week = ?",
#             (day_of_week,),
#         )
#         row = cursor.fetchone()
#         if row:
#             return row[0], row[1]
#         else:
#             return None, None
#     except sqlite3.Error as e:
#         print(f"Error fetching operating hours for {day_of_week}: {e}")
#         return None, None


# if __name__ == "__main__":
#     # This block will run only when this file is executed directly
#     conn = create_connection()
#     if conn:
#         create_menu_table(conn)
#         create_operating_hours_table(conn)
#         create_product_status_table(conn)

#         # Add some sample menu items
#         add_menu_item(
#             conn, "Sourdough Loaf", 5.00, "A classic tangy loaf with a crispy crust."
#         )
#         add_menu_item(
#             conn,
#             "Multigrain Bread",
#             6.50,
#             "A hearty and healthy loaf packed with various grains and seeds.",
#         )
#         add_menu_item(
#             conn,
#             "Croissant",
#             3.50,
#             "A buttery and flaky pastry with a golden-brown exterior.",
#         )
#         add_menu_item(
#             conn,
#             "Blueberry Scone",
#             4.50,
#             "A soft and crumbly scone filled with juicy blueberries.",
#         )

#         # Add some sample operating hours
#         add_operating_hours(conn, "Monday", "09:00", "17:00")
#         add_operating_hours(conn, "Tuesday", "09:00", "17:00")
#         add_operating_hours(conn, "Wednesday", "09:00", "17:00")
#         add_operating_hours(conn, "Thursday", "09:00", "17:00")
#         add_operating_hours(conn, "Friday", "10:00", "18:00")
#         add_operating_hours(conn, "Saturday", "10:00", "16:00")
#         add_operating_hours(conn, "Sunday", "Closed", "Closed")

#         # Fetch and print all menu items
#         menu_items = fetch_all_menu_items(conn)
#         if menu_items:
#             print("\n--- Current Menu ---")
#             for item in menu_items:
#                 print(
#                     f"Product: {item[0]}, Price: ${item[1]:.2f}, Description: {item[2]}"
#                 )

#         conn.close()
