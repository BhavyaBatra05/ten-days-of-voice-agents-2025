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

# Load company FAQ data
FAQ_FILE = Path("shared-data/day5_cashfree_faq.json")
LEADS_FILE = Path("shared-data/day5_leads.json")

def load_company_data():
    """Load company info, FAQs, and pricing from JSON file"""
    try:
        with open(FAQ_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading company data: {e}")
        return {"company_info": {}, "products": [], "pricing": {}, "faqs": []}

COMPANY_DATA = load_company_data()

# Lead tracking
class LeadTracker:
    """Track and manage lead information"""
    
    def __init__(self):
        self.data = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "timestamp": datetime.now().isoformat(),
            "questions_asked": []
        }
    
    def update_field(self, field: str, value: str):
        """Update a lead field"""
        if field in self.data:
            self.data[field] = value
            logger.info(f"Updated lead field {field}: {value}")
    
    def add_question(self, question: str):
        """Track questions asked during the conversation"""
        if question not in self.data["questions_asked"]:
            self.data["questions_asked"].append(question)
    
    def get_missing_fields(self):
        """Return list of fields that haven't been collected"""
        return [k for k, v in self.data.items() 
                if k not in ["timestamp", "questions_asked"] and v is None]
    
    def is_complete(self):
        """Check if all required fields are collected"""
        return len(self.get_missing_fields()) == 0
    
    def save_to_file(self):
        """Save lead data to single JSON file (append)"""
        # Load existing leads or create new list
        if LEADS_FILE.exists():
            with open(LEADS_FILE, "r") as f:
                all_leads = json.load(f)
        else:
            all_leads = []
        
        # Add new lead with timestamp
        self.data["timestamp"] = datetime.now().isoformat()
        all_leads.append(self.data)
        
        # Save back to file
        try:
            with open(LEADS_FILE, "w") as f:
                json.dump(all_leads, f, indent=2)
            logger.info(f"Lead saved to {LEADS_FILE} (total leads: {len(all_leads)})")
            return str(LEADS_FILE)
        except Exception as e:
            logger.error(f"Error saving lead: {e}")
            return None



class SDRAgent(Agent):
    """Sales Development Representative for Cashfree Payments - answers FAQs and captures leads"""
    
    def __init__(self) -> None:
        company_name = COMPANY_DATA.get("company_info", {}).get("name", "our company")
        company_tagline = COMPANY_DATA.get("company_info", {}).get("tagline", "")
        
        products_list = "\n".join([f"- {p['name']}: {p['description']}" for p in COMPANY_DATA.get("products", [])])
        
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting including emojis, asterisks, or other weird symbols.
            You are curious, friendly, and have a sense of humor.""",
        )



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Create main session - each agent will configure its own TTS
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8)
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
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
    
    # Start with SDR agent
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    await ctx.connect()




if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    

