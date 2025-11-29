import logging
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    UserInputTranscribedEvent,
    ConversationItemAddedEvent,
)
from livekit.agents.voice import room_io
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

load_dotenv(".env.local")


class GameMasterAgent(Agent):
    """D&D-style Game Master for interactive voice adventure (Day 8 primary goal)"""
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a Game Master running a fantasy adventure in a world of dragons and magic.
Your role:
- Describe the current scene in vivid detail.
- End every message with a prompt for player action (e.g., 'What do you do?').
- Remember the player's past decisions, named characters, and locations using chat history only.
- Keep the tone dramatic and immersive, but friendly.
- Guide the player through a short adventure (8-10 turns) with a complete mini-arc (finding something, escaping danger, defeating a foe, etc.).
- Pace the story so it reaches a satisfying conclusion within 10 turns.
- Do not break character or mention you are an AI.
""",
        )



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    logger.info("Connecting to room %s", ctx.room.name)
    
    
    # Create main session - each agent will configure its own TTS
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-ken",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8),
        ),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )
    # Metrics
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Start with Game Master agent
    @session.on("user_input_transcribed")
    def _on_user_input(ev: UserInputTranscribedEvent):
        logger.info(
            "USER TRANSCRIPT: %s (final=%s, lang=%s)",
            ev.transcript,
            ev.is_final,
            ev.language,
        )

    @session.on("conversation_item_added")
    def _on_conversation_item(ev: ConversationItemAddedEvent):
        # This gives you BOTH user + agent text as they are committed
        try:
            text = ev.item.text_content
        except Exception:
            text = str(ev.item)
        logger.info("ITEM ADDED: role=%s text=%s", ev.item.role, text)
    
    await session.start(
        room=ctx.room,
        agent=GameMasterAgent(),
        room_options=room_io.RoomOptions(
            # make sure **all of these** are enabled:
            audio_input=True,   # mic input to agent
            audio_output=True,  # agent TTS to room
            text_input=True,    # enables `lk.chat` / sendText()
            text_output=True,   # enables transcript stream -> LiveKit UI
        ),
    )
    
    await ctx.connect()
    
    await session.generate_reply(
        instructions=(
            "Start the fantasy adventure, introduce the setting and main hook, "
            "then ask the player what they do first."
        )
    )




if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


