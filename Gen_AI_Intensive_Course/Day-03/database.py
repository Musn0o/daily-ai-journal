import sqlite3

# Define the database file name
DATABASE_NAME = "Gen_AI_Intensive_Course/Day-03/data/scar_bakery.db"


def create_connection():
    """Creates a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        print(f"Connected to database: {DATABASE_NAME}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn


def create_menu_table(conn):
    """Creates the menu table in the database if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS menu (
                product_name TEXT PRIMARY KEY,
                price REAL,
                description TEXT
            )
        """)
        conn.commit()
        print("Menu table created successfully")
    except sqlite3.Error as e:
        print(f"Error creating menu table: {e}")


def create_operating_hours_table(conn):
    """Creates the operating_hours table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operating_hours (
                day_of_week TEXT PRIMARY KEY,
                opening_time TEXT,
                closing_time TEXT
            )
        """)
        conn.commit()
        print("Operating hours table created successfully")
    except sqlite3.Error as e:
        print(f"Error creating operating hours table: {e}")


def create_product_status_table(conn):
    """Creates the product_status table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_status (
                product_name TEXT PRIMARY KEY,
                status TEXT,
                availability_description TEXT,
                FOREIGN KEY (product_name) REFERENCES menu(product_name)
            )
        """)
        conn.commit()
        print("Product status table created successfully")
    except sqlite3.Error as e:
        print(f"Error creating product status table: {e}")


def add_menu_item(conn, product_name, price, description):
    """Adds a new menu item to the menu table."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO menu (product_name, price, description)
            VALUES (?, ?, ?)
        """,
            (product_name, price, description),
        )
        conn.commit()
        print(f"Added item: {product_name} to the menu")
    except sqlite3.Error as e:
        print(f"Error adding menu item: {e}")


def add_operating_hours(conn, day_of_week, opening_time, closing_time):
    """Adds operating hours for a specific day."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO operating_hours (day_of_week, opening_time, closing_time)
            VALUES (?, ?, ?)
        """,
            (day_of_week, opening_time, closing_time),
        )
        conn.commit()
        print(f"Added operating hours for {day_of_week}")
    except sqlite3.Error as e:
        print(f"Error adding operating hours for {day_of_week}: {e}")


def fetch_all_menu_items(conn):
    """Fetches all menu items from the menu table."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM menu")
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error fetching menu items: {e}")
        return None


def fetch_operating_hours_by_day(conn, day_of_week):
    """Fetches the operating hours for a specific day of the week."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT opening_time, closing_time FROM operating_hours WHERE day_of_week = ?",
            (day_of_week,),
        )
        row = cursor.fetchone()
        if row:
            return row[0], row[1]
        else:
            return None, None
    except sqlite3.Error as e:
        print(f"Error fetching operating hours for {day_of_week}: {e}")
        return None, None


if __name__ == "__main__":
    # This block will run only when this file is executed directly
    conn = create_connection()
    if conn:
        create_menu_table(conn)
        create_operating_hours_table(conn)
        create_product_status_table(conn)

        # Add some sample menu items
        add_menu_item(
            conn, "Sourdough Loaf", 5.00, "A classic tangy loaf with a crispy crust."
        )
        add_menu_item(
            conn,
            "Multigrain Bread",
            6.50,
            "A hearty and healthy loaf packed with various grains and seeds.",
        )
        add_menu_item(
            conn,
            "Croissant",
            3.50,
            "A buttery and flaky pastry with a golden-brown exterior.",
        )
        add_menu_item(
            conn,
            "Blueberry Scone",
            4.50,
            "A soft and crumbly scone filled with juicy blueberries.",
        )

        # Add some sample operating hours
        add_operating_hours(conn, "Monday", "09:00", "17:00")
        add_operating_hours(conn, "Tuesday", "09:00", "17:00")
        add_operating_hours(conn, "Wednesday", "09:00", "17:00")
        add_operating_hours(conn, "Thursday", "09:00", "17:00")
        add_operating_hours(conn, "Friday", "10:00", "18:00")
        add_operating_hours(conn, "Saturday", "10:00", "16:00")
        add_operating_hours(conn, "Sunday", "Closed", "Closed")

        # Fetch and print all menu items
        menu_items = fetch_all_menu_items(conn)
        if menu_items:
            print("\n--- Current Menu ---")
            for item in menu_items:
                print(
                    f"Product: {item[0]}, Price: ${item[1]:.2f}, Description: {item[2]}"
                )

        conn.close()
