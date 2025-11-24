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

# Global wellness session state
current_session = {
    "date": None,
    "mood": None,
    "energy": None,
    "stress": None,
    "objectives": [],
    "summary": None
}


class Assistant(Agent):
    def __init__(self) -> None:
        # Load previous check-ins to provide context
        self.previous_context = self._load_previous_sessions()
        
        super().__init__(
            instructions=f"""You are a supportive Health and Wellness Voice Companion. You conduct daily check-ins with users to help them reflect on their wellbeing and set intentions.

            Your role is to:
            - Be warm, empathetic, and non-judgmental
            - Ask about mood, energy levels, and stress
            - Help users identify 1-3 daily objectives or intentions
            - Offer simple, practical, grounded advice (no medical diagnosis)
            - Keep responses conversational and natural
            
            IMPORTANT: You are NOT a medical professional. Avoid diagnosis or medical claims. Focus on supportive conversation and practical suggestions.
            
            Check-in Flow:
            1. Start by introducing yourself: "Hi! I'm your Health and Wellness Companion from Cult.fit." Then warmly ask about their mood and energy today
            2. Ask about any stress or concerns
            3. Help them identify 1-3 objectives or intentions for today
            4. Offer a small piece of practical advice or reflection
            5. Recap the key points and confirm with them
            6. MUST call save_wellness_session tool with a brief summary to save the check-in. Do not end the conversation without calling this tool.
            
            Previous Context:
            {self.previous_context}
            
            Use the previous context to make connections: "Last time we talked, you mentioned... How does today compare?"
            
            Keep responses short and conversational, as if you're a supportive friend checking in.""",
        )
    
    def _load_previous_sessions(self) -> str:
        """Load the last 3 check-ins to provide context"""
        wellness_file = Path("wellness_log.json")
        
        if not wellness_file.exists():
            return "This is our first conversation together."
        
        try:
            with open(wellness_file, "r") as f:
                sessions = json.load(f)
            
            if not sessions:
                return "This is our first conversation together."
            
            # Get last 3 sessions
            recent = sessions[-3:]
            context_parts = []
            
            for session in recent:
                date = session.get("date", "Unknown date")
                mood = session.get("mood", "not specified")
                energy = session.get("energy", "not specified")
                objectives = session.get("objectives", [])
                
                context_parts.append(
                    f"On {date}: Mood was {mood}, energy was {energy}. "
                    f"Objectives: {', '.join(objectives) if objectives else 'none set'}"
                )
            
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error loading previous sessions: {e}")
            return "This is our first conversation together."

    @function_tool
    async def update_wellness_session(
        self,
        context: RunContext,
        mood: str | None = None,
        energy: str | None = None,
        stress: str | None = None,
        objectives: list[str] | None = None
    ):
        """Update the current wellness check-in session with user information.
        
        Args:
            mood: How the user is feeling (e.g., happy, tired, anxious, good, etc.)
            energy: User's energy level (e.g., high, low, medium, drained, energized)
            stress: Any stressors or concerns (text description)
            objectives: List of 1-3 things the user wants to accomplish today
        """
        global current_session
        
        if mood:
            current_session["mood"] = mood
        if energy:
            current_session["energy"] = energy
        if stress:
            current_session["stress"] = stress
        if objectives:
            current_session["objectives"] = objectives
        
        current_session["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Session updated: {current_session}")
        
        # Check what's still needed
        missing = []
        if not current_session["mood"]:
            missing.append("mood")
        if not current_session["energy"]:
            missing.append("energy level")
        if not current_session["objectives"] or len(current_session["objectives"]) == 0:
            missing.append("objectives for today")
        
        if missing:
            return f"Got it. Still need to know about: {', '.join(missing)}"
        else:
            return "Perfect! I have a good understanding of how you're doing today and what you want to accomplish."

    @function_tool
    async def save_wellness_session(self, context: RunContext, summary: str):
        """Save the completed wellness check-in to the log file. Call this after recapping with the user.
        
        Args:
            summary: A brief summary sentence about this check-in. ALWAYS use present tense. Format: "User is feeling [mood], energy is [level], and [stress situation]. Objectives are [list objectives]."
        """
        global current_session
        
        # Verify session is complete
        if not all([
            current_session["mood"],
            current_session["energy"],
            current_session["objectives"]
        ]):
            return "Cannot save session - missing required information (mood, energy, or objectives)"
        
        current_session["summary"] = summary
        
        # Load existing log
        wellness_file = Path("wellness_log.json")
        
        if wellness_file.exists():
            with open(wellness_file, "r") as f:
                sessions = json.load(f)
        else:
            sessions = []
        
        # Add current session
        sessions.append(current_session.copy())
        
        # Save back to file
        with open(wellness_file, "w") as f:
            json.dump(sessions, f, indent=2)
        
        logger.info(f"Wellness session saved: {current_session}")
        
        # Reset for next session
        for key in current_session:
            if key == "objectives":
                current_session[key] = []
            else:
                current_session[key] = None
        
        return "Check-in saved successfully! Looking forward to our next conversation. Take care!"


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
