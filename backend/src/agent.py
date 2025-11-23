import logging
import json
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Global order state
current_order = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly barista at a coffee shop. Greet customers warmly and help them place their orders.
            
            When a customer wants to order, mention our menu: "We have cappuccino, latte, americano, espresso, mocha, flat white, and macchiato."
            
            You need to collect the following information for each order:
            - Drink type (cappuccino, latte, americano, espresso, mocha, flat white, macchiato, etc.)
            - Size: For ESPRESSO ask "single or double shot?", for all other drinks ask "small, medium, or large?"
            - Milk type (whole, skim, oat, almond, soy, or none)
            - Any extras (whipped cream, extra shot, caramel drizzle, vanilla syrup, etc.)
            - Customer's name for the order
            
            IMPORTANT RULES:
            - For ESPRESSO: Ask "single or double shot?" instead of size
            - For LATTE, CAPPUCCINO, FLAT WHITE, MOCHA: These REQUIRE milk. If customer says "no milk", politely explain that these drinks need milk, or suggest an Americano or Espresso instead
            - For AMERICANO: Milk is optional (just a splash). Default to "none" if not mentioned
            - Ask for ONE piece of information at a time. Wait for the customer's response before asking the next question
            
            Always be conversational and ask clarifying questions if information is missing.
            Once you have all the information, use the save_order tool to complete the order.
            Keep responses short and natural, as if you're having a real conversation at a coffee counter.""",
        )

    @function_tool
    async def update_order(
        self,
        context: RunContext,
        drink_type: str | None = None,
        size: str | None = None,
        milk: str | None = None,
        extras: list[str] | None = None,
        name: str | None = None
    ):
        """Update the current order with customer information. Use this to track what the customer wants.
        
        Args:
            drink_type: Type of drink (coffee, latte, cappuccino, etc.)
            size: Size of drink (small, medium, large) OR for espresso: single/double shot
            milk: Type of milk (whole, skim, oat, almond, soy, none)
            extras: List of extras (whipped cream, extra shot, vanilla syrup, etc.)
            name: Customer's name for the order
        """
        global current_order
        
        if drink_type:
            current_order["drinkType"] = drink_type.lower()
            
        if size:
            current_order["size"] = size.lower()
            
        if milk:
            milk_lower = milk.lower()
            drink = current_order["drinkType"]
            
            # Validate milk for drinks that require it
            if drink in ["latte", "cappuccino", "flat white", "mocha"]:
                if milk_lower == "none" or milk_lower == "no":
                    return f"A {drink} requires milk. Would you like to choose a milk type, or would you prefer an Americano or Espresso instead?"
            
            # For Americano, default to none if not specified
            if drink == "americano" and milk_lower == "none":
                current_order["milk"] = "none"
            else:
                current_order["milk"] = milk_lower
                
        if extras:
            current_order["extras"] = extras
        if name:
            current_order["name"] = name
            
        logger.info(f"Order updated: {current_order}")
        
        # Check if order is complete
        missing = []
        if not current_order["drinkType"]:
            missing.append("drink type")
        if not current_order["size"]:
            drink = current_order["drinkType"]
            if drink == "espresso":
                missing.append("shot size (single or double)")
            else:
                missing.append("size")
        if current_order["milk"] is None:
            missing.append("milk preference")
        if not current_order["name"]:
            missing.append("name")
            
        if missing:
            return f"Got it. Still need: {', '.join(missing)}"
        else:
            return "Perfect! I have everything I need for your order."

    @function_tool
    async def save_order(self, context: RunContext):
        """Save the completed order to a JSON file. Only call this when all order fields are filled.
        """
        global current_order
        
        # Verify order is complete
        if not all([
            current_order["drinkType"],
            current_order["size"],
            current_order["milk"] is not None,
            current_order["name"]
        ]):
            return "Cannot save order - missing information"
        
        # Save to JSON file
        orders_file = Path("orders.json")
        
        # Load existing orders or create new list
        if orders_file.exists():
            with open(orders_file, "r") as f:
                orders = json.load(f)
        else:
            orders = []
        
        # Add current order
        orders.append(current_order.copy())
        
        # Save back to file
        with open(orders_file, "w") as f:
            json.dump(orders, f, indent=2)
        
        logger.info(f"Order saved: {current_order}")
        
        # Reset for next order
        for key in current_order:
            if key == "extras":
                current_order[key] = []
            else:
                current_order[key] = None
        
        return "Order saved successfully! Your drink will be ready shortly."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
