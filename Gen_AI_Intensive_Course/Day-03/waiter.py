import os
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from collections.abc import Iterable
from random import randint
from langchain_core.messages.tool import ToolMessage
from database import (
    create_connection,
    fetch_all_menu_items,
    fetch_operating_hours_by_day,
)
import datetime
from designer import MenuDesigner  # Import the class

# ... other imports you might have ...
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the MenuDesigner
menu_designer = MenuDesigner(GEMINI_API_KEY)


class OrderState(TypedDict):
    """State representing the customer's order conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress order.
    order: list[str]

    # Flag indicating that the order is placed and completed.
    finished: bool


# The system instruction defines how the chatbot is expected to behave and includes
# rules for when to call different functions, as well as rules for the conversation, such
# as tone and what is permitted for discussion.
WAITERBOT_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a WaiterBot, an interactive Bakery ordering system in Scar's Bakery. A human will talk to you about the "
    "available products you have and you will answer any questions about menu items (and only about "
    "menu items - no off-topic discussion, but you can chat about the products and their history). "
    "The customer will place an order for 1 or more items from the menu, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user) "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with baked goods and modifier names from the MENU before adding them to the order. "
    "If you are unsure a baked goods or modifier matches those on the MENU, ask a question to clarify or redirect. "
    "You only have the modifiers listed on the menu. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
    "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!"
    "\n\n"
    "If any of the tools are unavailable, you can break the fourth wall and tell the user that "
    "they have not implemented them yet and should keep reading to do so.",
)

image_path = menu_designer.generate_menu_image()

if image_path:
    WELCOME_MSG = f"Welcome to Scar's Bakery.\n\n{image_path}\n\nType `Bye` to quit. How may I serve you today?"
else:
    WELCOME_MSG = (
        "Welcome to Scar's Bakery. Type `Bye` to quit. How may I serve you today?"
    )


@tool
def get_menu() -> str:
    """Provide the latest up-to-date bakery menu image and text."""
    now = datetime.datetime.now()
    # This will give you the full day name (e.g., "Monday")
    day_of_week = now.strftime("%A")
    conn = create_connection()

    if conn:
        opening_time, closing_time = fetch_operating_hours_by_day(conn, day_of_week)
        menu_items_from_db = fetch_all_menu_items(conn)
    else:
        opening_time, closing_time = None, None

    operating_hours_message = ""
    if opening_time and closing_time:
        if opening_time == "Closed" and closing_time == "Closed":
            operating_hours_message = (
                f"📢 Today is {day_of_week}, and we are currently closed. 😴"
            )
        else:
            operating_hours_message = f"📢 Welcome to Scar's Bakery! Today's hours ({day_of_week}): {opening_time} - {closing_time}."
    else:
        operating_hours_message = f"📢 Welcome to Scar's Bakery! Today's hours for {day_of_week} are not currently available."

    menu = {}
    if menu_items_from_db:
        for item in menu_items_from_db:
            product_name, price, description = item
            menu[product_name] = {"price": price, "description": description}

    operating_hours_info = {}
    if opening_time and closing_time:
        operating_hours_info["opening_time"] = opening_time
        operating_hours_info["closing_time"] = closing_time
        operating_hours_info["status"] = (
            "Closed" if opening_time == "Closed" else "Open"
        )
    else:
        operating_hours_info["status"] = "Hours not available"

    if conn:
        conn.close()

        menu_data = {
            "operating_hours": operating_hours_info,
            "menu": menu,
            "operating_hours_text": operating_hours_message,
        }

    return menu_data


# @tool
# def get_menu_image():
#     """Generate and provide the latest up-to-date bakery menu image."""
#     menu_data = get_menu()  # Get the menu data
#     operating_hours_text = menu_data.get("operating_hours_text", "")
#     menu_items = menu_data.get("menu", {})

#     # Call the image generation function from our MenuDesigner class
#     image_path = menu_designer.generate_menu_image(menu_items, operating_hours_text)

#     if image_path:
#         return f"Here's the menu image: {image_path}"  # For now, just return the path
#     else:
#         return "Sorry, I couldn't generate the menu image right now."


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([WAITERBOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}


def human_node(state: OrderState) -> OrderState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Waiter:", last_msg.content)

    user_input = input("Customer: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"bye", "Bye", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}


def order_node(state: OrderState) -> OrderState:
    """The ordering node. This is where the order state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":
            # Each order item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            order.append(f"{tool_call['args']['item']} ({modifier_str})")
            response = "\n".join(order)

        elif tool_call["name"] == "confirm_order":
            # We could entrust the LLM to do order confirmation, but it is a good practice to
            # show the user the exact data that comprises their order so that what they confirm
            # precisely matches the order that goes to the kitchen - avoiding hallucination
            # or reality skew.

            # In a real scenario, this is where you would connect your POS screen to show the
            # order to the user.

            print("Your order:")
            if not order:
                print("  (no items)")

            for item in order:
                print(f"  {item}")

            response = input("Is this correct? ")

        elif tool_call["name"] == "get_order":
            response = "\n".join(order) if order else "(no order)"

        elif tool_call["name"] == "clear_order":
            order.clear()
            response = None

        elif tool_call["name"] == "place_order":
            order_text = "\n".join(order)
            print("Sending order to kitchen!")
            print(order_text)

            # TODO(you!): Implement bakery order processing.
            order_placed = True
            response = randint(1, 5)  # ETA in minutes

        else:
            raise NotImplementedError(f"Unknown tool call: {tool_call['name']}")

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": order_placed}


def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if state.get("finished", False):
        # When an order is placed, exit the app. The system instruction indicates
        # that the chatbot should say thanks and goodbye at this point, so we can exit
        # cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "ordering"

    else:
        return "human"


def maybe_exit_human_node(state: OrderState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"


@tool
def add_to_order(item: str, modifiers: Iterable[str]) -> str:
    """Adds the specified baked goods to the customer's order, including any modifiers.

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
def place_order() -> int:
    """Sends the order to the waiter for fulfillment.

    Returns:
      The estimated number of minutes until the order is ready.
    """


# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_menu]
tool_node = ToolNode(auto_tools)

# Order-tools will be handled by the order node.
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]

# The LLM needs to know about all of the tools, so specify everything here.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


graph_builder = StateGraph(OrderState)

# Nodes
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

# Chatbot -> {ordering, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools (both kinds) always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_order_tools = graph_builder.compile()

config = {"recursion_limit": 100}

state = graph_with_order_tools.invoke({"messages": []}, config)
