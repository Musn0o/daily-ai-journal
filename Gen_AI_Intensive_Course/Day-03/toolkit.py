from langchain_core.tools import tool
from database_utils import get_menu_items
from langgraph.prebuilt import ToolNode


@tool
def get_menu() -> dict:
    """Provide the latest up-to-date bakery menu."""
    return get_menu_items()


@tool
def add_to_order(item: str) -> str:
    """Adds the specified baked goods to the customer's order.

    Returns:
      The updated order in progress.
    """


@tool
def confirm_order() -> str:
    """Asks the customer if the order is correct.

    Returns:
      The user's free-text response.
    """


@tool
def get_order() -> str:
    """Returns the users order so far. One item per line."""


@tool
def clear_order():
    """Removes all items from the user's order."""


@tool
def place_order() -> str:
    """Sends the order to the kitchen for fulfillment and update stock.

    Returns:
      The estimated number of minutes until the order is ready.
    """


@tool
def get_special_dish() -> str:
    """Use this function if the user asks about the today's special, a recommended dish, or Scar's best dish for the day."""


@tool
def get_customer_name(name: str) -> str:
    """Captures the customer's name."""
    return f"Customer name received: {name}"


@tool
def get_customer_email(email: str) -> str:
    """Captures the customer's email address."""
    return f"Customer email received: {email}"


# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)

# Order-tools will be handled by the order node.
order_tools = [
    add_to_order,
    confirm_order,
    get_order,
    clear_order,
    place_order,
    # get_special_dish,
    get_customer_name,
    get_customer_email,
]
