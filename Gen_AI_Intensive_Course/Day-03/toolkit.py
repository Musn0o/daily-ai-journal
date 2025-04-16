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
    """Sends the order to the waiter for fulfillment.

    Returns:
      The estimated number of minutes until the order is ready.
    """


@tool
def get_special_dish() -> str:
    """Use this function if the user asks about the today's special, a recommended dish, or Scar's best dish for the day."""


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
    get_special_dish,
]
