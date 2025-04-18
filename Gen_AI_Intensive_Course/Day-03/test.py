import os
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from random import randint
from langchain_core.messages.tool import ToolMessage
from database_utils import process_bakery_order, create_connection
from toolkit import order_tools, auto_tools, tool_node
from bakery_beat import BakeryBeat
from administrator import Administrator
from prompts import WAITERBOT_SYSINT

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# # Initialize the MenuDesigner
# chronos_companion = BakeryBeat()
# bakery_designer = Designer()
# bakery_designer = Designer()


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


class Waiter:
    def __init__(self):
        self.DATABASE_NAME = "/media/scar/HDD_Data/Repositories/daily-ai-journal/Gen_AI_Intensive_Course/Day-03/data/scar_bakery_ai.db"
        self.conn = create_connection()  # Use your create_connection here if you prefer
        self.cursor = self.conn.cursor()
        self.chronos_companion = BakeryBeat()  # Pass the connection here
        # self.bakery_designer = Designer()

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def chatbot_with_tools(
        self, state: OrderState
    ) -> OrderState:  # Removed @staticmethod
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        defaults = {
            "order": [],
            "finished": False,
            "is_admin_mode": False,
            "admin_initiated": False,
        }
        if state["messages"]:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

            llm_with_tools = llm.bind_tools(auto_tools + order_tools)
            new_output = llm_with_tools.invoke([WAITERBOT_SYSINT] + state["messages"])
        else:
            new_output = AIMessage(
                content=self.chronos_companion.generate_welcome_message()
            )

        # Set up some defaults if not already set, then pass through the provided state,
        # overriding only the "messages" field.
        return defaults | state | {"messages": [new_output]}

    def human_node(self, state: OrderState) -> OrderState:  # Removed @staticmethod
        """Display the last model message to the user, and receive the user's input."""
        last_msg = state["messages"][-1]
        print("Waiter:", last_msg.content)

        user_input = input("Customer: ")

        if user_input == "!Scar_in":
            print("Admin mode activation initiated.")
            supervisor = Administrator()
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

    def route_after_human_node(
        self, state: OrderState
    ) -> dict:  # Removed @staticmethod
        """Routes to chatbot or end based on the finished flag."""
        if state.get("finished", False):
            return {"next": END}
        else:
            return {"next": "chatbot"}

    def order_node(self, state: OrderState) -> OrderState:  # Removed @staticmethod
        """The ordering node. This is where the order state is manipulated."""
        print("--- Entering order_node ---")  # Added log
        print(f"Current state in order_node: {state}")  # Added log
        tool_msg = state.get("messages", [])[-1]
        order = state.get("order", [])
        outbound_msgs = []
        order_placed = False

        for tool_call in tool_msg.tool_calls:
            if tool_call["name"] == "add_to_order":
                order.append(f"{tool_call['args']['item']}")
                response = "\n".join(order)

            elif tool_call["name"] == "confirm_order":
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

                # Assuming you have customer_name and customer_email variables available here
                customer_name = "Scar"  # Replace with the actual customer name
                customer_email = (
                    "scar@example.com"  # Replace with the actual customer email
                )

                order_placed = process_bakery_order(
                    customer_name, customer_email, order
                )
                print(
                    f"Value of order_placed after calling process_bakery_order: {order_placed}"
                )  # Added log
                if order_placed:
                    response = randint(1, 5)  # ETA in minutes
                else:
                    response = (
                        "There was an issue processing your order. Please try again."
                    )

            # elif tool_call["name"] == "get_special_dish":
            #     special_product = "Chocolate Croissant"  # Let's hardcode it for now
            #     image_path = self.bakery_designer.generate_special_dish_image(
            #         special_product
            #     )
            #     response_message = f"Great choice! Here's a look at our special '{special_product}': {image_path}"
            #     print(response_message)
            #     response = (
            #         response_message  # Assign the message to the response variable
            #     )

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

        # print(
        #     f"State being returned from order_node: {state | {{'messages': outbound_msgs, 'order': order[0], 'finished': order_placed}}}"
        # )  # Added log
        print("--- Exiting order_node ---")  # Added log
        return {"messages": outbound_msgs, "order": order, "finished": order_placed}

    def maybe_route_to_tools(self, state: OrderState) -> str:  # Removed @staticmethod
        """Route between chat and tool nodes if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        msg = msgs[-1]

        if state.get("finished", False):
            return END

        elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            if any(
                tool["name"] in tool_node.tools_by_name.keys()
                for tool in msg.tool_calls
            ):
                return "tools"
            else:
                return "ordering"

        else:
            return "human"

    def maybe_exit_human_node(
        self, state: OrderState
    ) -> Literal["chatbot", "__end__"]:  # Removed @staticmethod
        """Route to the chatbot, unless it looks like the user is exiting."""
        if state.get("finished", False):
            return END
        else:
            return "chatbot"

    def check_opening_hours_node(self, state: OrderState):  # Removed @staticmethod
        """Checks if the bakery is open and returns a routing decision in a dictionary."""
        if self.chronos_companion.is_bakery_open():
            return {"next": "open"}
        else:
            print(
                "Schedule Buddy: Aww, so sorry! 😔 Scar's Bakery is currently closed for the day. We'll be back bright and early to bake you more delicious treats! 🥐 See you soon!"
            )
            return {"next": "closed"}

    def say_closed_node(self, state: OrderState):  # Removed @staticmethod
        """Informs the customer that the bakery is closed."""
        return {
            "messages": [
                AIMessage(
                    content="I'm very sorry, but Scar's Bakery is closed for today. Please come back another time!"
                )
            ]
        }

    def initiate_admin_mode_node(
        self, state: OrderState
    ) -> OrderState:  # Removed @staticmethod
        """Initiates the admin mode by interacting with the BakerySupervisor and sets the finished flag."""
        supervisor = Administrator()
        supervisor.start_admin_mode()
        print("Admin mode initiated in BakerySupervisor.")
        return state | {"finished": True}

    def supervisor_interaction_node(
        self, state: OrderState
    ) -> OrderState:  # Removed @staticmethod
        print("Now in supervisor interaction phase (from waiter.py).")
        return state

    def run(self):  # Create a run method to encapsulate the graph setup and invocation
        graph_builder = StateGraph(OrderState)

        # Nodes
        graph_builder.add_node("check_opening_hours", self.check_opening_hours_node)
        graph_builder.add_node("chatbot", self.chatbot_with_tools)
        graph_builder.add_node("human", self.human_node)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("ordering", self.order_node)
        graph_builder.add_node("end_closed", self.say_closed_node)
        graph_builder.add_node("route_after_human", self.route_after_human_node)

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
        graph_builder.add_conditional_edges("chatbot", self.maybe_route_to_tools)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("ordering", "chatbot")
        graph_builder.add_edge("end_closed", END)
        graph_builder.add_edge(START, "check_opening_hours")

        graph_with_order_tools = graph_builder.compile()

        config = {"recursion_limit": 100}

        state = graph_with_order_tools.invoke({"messages": []}, config)
        self.close_connection()  # Close the connection when the interaction finishes


if __name__ == "__main__":
    waiter_bot = Waiter()  # Instantiate the Waiter class
    waiter_bot.run()  # Run the LangGraph flow through the instance
