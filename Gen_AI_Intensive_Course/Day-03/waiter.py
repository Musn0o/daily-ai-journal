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
from database_utils import (
    create_connection,
    fetch_all_menu_items,
)
from bakery_beat import BakeryBeat
from database_manager import BakerySupervisor
from designer import Designer
from prompts import WAITERBOT_SYSINT

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the MenuDesigner
chronos_companion = BakeryBeat()
supervisor = BakerySupervisor()
bakery_designer = Designer()


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

    is_admin_mode: bool
    in_admin_interaction: bool


@tool
def get_menu() -> dict:
    """Provide the latest up-to-date bakery menu."""
    # TODO: Refactor to get menu from supervisor.supervisor.get_menu()
    conn = create_connection()
    menu = {}
    if conn:
        menu_items_from_db = fetch_all_menu_items(conn)
        for item in menu_items_from_db:
            product_name, price, description = item
            menu[product_name] = {"price": price, "description": description}
        conn.close()
    return {"menu": menu}


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {
        "order": [],
        "finished": False,
        "is_admin_mode": False,
        "admin_initiated": False,
    }
    if state["messages"]:
        new_output = llm_with_tools.invoke([WAITERBOT_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=chronos_companion.generate_welcome_message())

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}


def human_node(state: OrderState) -> OrderState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Waiter:", last_msg.content)

    user_input = input("Customer: ")

    if user_input == "!Scar_in":
        print("Admin mode activation initiated.")
        supervisor = BakerySupervisor()
        supervisor.start_admin_mode()
        print("Admin mode initiated in BakerySupervisor.")
        return state | {
            "is_admin_mode": True,
            "admin_initiated": True,
            "finished": True,
            "messages": state["messages"] + [("user", user_input)],
        }
    elif user_input in {"bye", "Bye", "quit", "exit", "goodbye"}:
        return state | {
            "finished": True,
            "messages": state["messages"] + [("user", user_input)],
        }
    else:
        return state | {"messages": [("user", user_input)]}


def route_after_human_node(state: OrderState) -> dict:
    """Routes to chatbot or end based on the finished flag."""
    if state.get("finished", False):
        return {"next": END}
    else:
        return {"next": "chatbot"}


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

        elif tool_call["name"] == "get_special_dish":
            special_product = "Chocolate Croissant"  # Let's hardcode it for now
            image_path = bakery_designer.generate_special_dish_image(
                special_product="Chocolate Croissant"
            )
            response_message = f"Great choice! Here's a look at our special '{special_product}': {image_path}"
            print(response_message)
            response = response_message  # Assign the message to the response variable

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


def check_opening_hours_node(state: OrderState):
    """Checks if the bakery is open and returns a routing decision in a dictionary."""
    if chronos_companion.is_bakery_open():
        return {"next": "open"}
    else:
        print(
            "Schedule Buddy: Aww, so sorry! 😔 Scar's Bakery is currently closed for the day. We'll be back bright and early to bake you more delicious treats! 🥐 See you soon!"
        )
        return {"next": "closed"}


def say_closed_node(state: OrderState):
    """Informs the customer that the bakery is closed."""
    return {
        "messages": [
            AIMessage(
                content="I'm very sorry, but Scar's Bakery is closed for today. Please come back another time!"
            )
        ]
    }


def initiate_admin_mode_node(state: OrderState) -> OrderState:
    """Initiates the admin mode by interacting with the BakerySupervisor and sets the finished flag."""
    supervisor = BakerySupervisor()
    supervisor.start_admin_mode()
    print("Admin mode initiated in BakerySupervisor.")
    return state | {"finished": True}


def supervisor_interaction_node(state: OrderState) -> OrderState:
    print("Now in supervisor interaction phase (from waiter.py).")
    return state


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

# The LLM needs to know about all of the tools, so specify everything here.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)


graph_builder = StateGraph(OrderState)

# Nodes
graph_builder.add_node("check_opening_hours", check_opening_hours_node)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)
graph_builder.add_node("end_closed", say_closed_node)
graph_builder.add_node("route_after_human", route_after_human_node)

# Edges
graph_builder.add_conditional_edges(
    "check_opening_hours",
    lambda state: state.get("next"),
    {
        "open": "chatbot",
        "closed": "end_closed",
    },
)
graph_builder.add_conditional_edges(
    "human",
    lambda state: "end_session"
    if state.get("admin_initiated", False)
    else "regular_flow",
    {
        "regular_flow": "route_after_human",
        "end_session": END,
    },
)
graph_builder.add_conditional_edges(
    "route_after_human",
    lambda state: state.get("next"),
    {
        "chatbot": "chatbot",
        END: END,
    },
)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")
graph_builder.add_edge("end_closed", END)
graph_builder.add_edge(START, "check_opening_hours")

graph_with_order_tools = graph_builder.compile()

config = {"recursion_limit": 100}

state = graph_with_order_tools.invoke({"messages": []}, config)
