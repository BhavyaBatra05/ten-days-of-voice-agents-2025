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
            instructions=f"""You are a warm and helpful Sales Development Representative for {company_name} - {company_tagline}.

Our products: {products_list}

MANDATORY TOOL USAGE - YOU MUST USE THESE TOOLS:
When user tells you their name → IMMEDIATELY call update_lead_name tool with their name
When user tells you company → IMMEDIATELY call update_lead_company tool
When user tells you email → IMMEDIATELY call update_lead_email tool
When user tells you role → IMMEDIATELY call update_lead_role tool
When user tells you use case → IMMEDIATELY call update_lead_use_case tool
When user tells you team size → IMMEDIATELY call update_lead_team_size tool
When user tells you timeline → IMMEDIATELY call update_lead_timeline tool
When user asks product question → IMMEDIATELY call search_faq tool

CONVERSATION FLOW (FOLLOW EXACTLY):
1. User shares need → Call update_lead_use_case → Say: "Got it! What's your name?"
2. User says name → Call update_lead_name → Say: "Thanks! What company?"
3. User says company → Call update_lead_company → Say: "Great! Your email?"
4. User says email → Call update_lead_email → Say: "Perfect! Your role?"
5. User says role → Call update_lead_role → Say: "Nice! Team size?"
6. User says team size → Call update_lead_team_size → Say: "Got it! Timeline?"
7. User says timeline → Call update_lead_timeline → Say: "Perfect! Any questions?"
8. User asks question → Call search_faq → Answer briefly → Ask: "Other questions?"
9. User says "no" or "that's all" → Call end_call

SPEED RULES:
- Tools return "saved" - ignore it, just ask next question immediately
- Example: User says "Bhavya" → [call tool] → "Thanks! What company?" (FAST!)
- Keep responses to 3-5 words during lead collection
- Don't repeat information back - just acknowledge and move on
- The faster you ask, the better

BE EXTREMELY FAST - NO LONG PAUSES!
""",
            tts=murf.TTS(
                voice="en-IN-priya",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8)
            )
        )
        self.lead_tracker = LeadTracker()
    
    async def on_enter(self) -> None:
        """Called when agent starts"""
        company_name = COMPANY_DATA.get("company_info", {}).get("name", "our company")
        await self.session.generate_reply(
            instructions=f"Greet the visitor warmly. Introduce yourself as an SDR from {company_name}. Ask what brought them here today and what they're working on."
        )
    
    @function_tool
    async def search_faq(self, context: RunContext, query: str):
        """Search FAQs for relevant information about products, pricing, or features
        
        Args:
            query: The question or topic to search for (e.g., "pricing", "payment gateway", "settlement time")
        """
        query_lower = query.lower()
        faqs = COMPANY_DATA.get("faqs", [])
        
        # Extract keywords from query (split and clean)
        query_words = set(word.strip("?,.:;!") for word in query_lower.split() if len(word) > 3)
        
        # Score-based search - find most relevant FAQs
        scored_matches = []
        for faq in faqs:
            question = faq.get("question", "").lower()
            answer = faq.get("answer", "").lower()
            category = faq.get("category", "")
            
            score = 0
            # Exact phrase match gets highest score
            if query_lower in question:
                score += 10
            if query_lower in answer:
                score += 5
            
            # Word-based matching
            question_words = set(question.split())
            answer_words = set(answer.split())
            
            for word in query_words:
                if word in question_words:
                    score += 3
                if word in answer_words:
                    score += 1
            
            if score > 0:
                scored_matches.append({
                    "question": faq.get("question"),
                    "answer": faq.get("answer"),
                    "category": category,
                    "score": score
                })
        
        # Sort by score (highest first)
        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Track the question
        self.lead_tracker.add_question(query)
        
        if scored_matches:
            # Return top 3 most relevant matches
            top_matches = scored_matches[:3]
            result = f"Here's what I found:\n\n"
            for i, match in enumerate(top_matches, 1):
                result += f"{i}. Q: {match['question']}\n   A: {match['answer']}\n\n"
            return result
        
        # If no FAQ matches, search in products and pricing
        products = COMPANY_DATA.get("products", [])
        for product in products:
            product_name = product.get("name", "").lower()
            product_desc = product.get("description", "").lower()
            
            # Check if any query word matches product
            if any(word in product_name or word in product_desc for word in query_words):
                features = product.get("features", [])
                feature_text = "\nKey Features:\n" + "\n".join(f"- {f}" for f in features) if features else ""
                return f"About {product.get('name')}:\n{product.get('description')}{feature_text}"
        
        # Search in pricing if query contains price-related keywords
        price_keywords = {"price", "pricing", "cost", "fee", "charge", "rate", "payment"}
        if any(keyword in query_lower for keyword in price_keywords):
            pricing = COMPANY_DATA.get("pricing", {})
            gateway_pricing = pricing.get("payment_gateway", {})
            if gateway_pricing:
                return f"""Here's our pricing information:

Payment Gateway:
- Domestic cards: {gateway_pricing.get('domestic_cards', 'N/A')}
- International cards: {gateway_pricing.get('international_cards', 'N/A')}
- UPI: {gateway_pricing.get('upi', 'N/A')}
- Netbanking: {gateway_pricing.get('netbanking', 'N/A')}
- Wallets: {gateway_pricing.get('wallets', 'N/A')}

No setup fee or annual fee!

Payment Links and Pages follow the same pricing as the gateway."""
        
        # Last resort - return general company info
        company_info = COMPANY_DATA.get("company_info", {})
        return f"""I'd be happy to help! Let me tell you about {company_info.get('name', 'us')}:

{company_info.get('description', '')}

We serve: {company_info.get('target_customers', 'businesses across India')}

Feel free to ask me about:
- Our products (Payment Gateway, Payment Links, Subscriptions, etc.)
- Pricing and fees
- Technical integration
- Settlements and payouts
- Security and compliance

What would you like to know more about?"""
    
    @function_tool
    async def update_lead_name(self, context: RunContext, name: str):
        """Save the lead's name
        
        Args:
            name: The person's full name
        """
        self.lead_tracker.update_field("name", name)
        return "saved"
    
    @function_tool
    async def update_lead_company(self, context: RunContext, company: str):
        """Save the lead's company name
        
        Args:
            company: The company name
        """
        self.lead_tracker.update_field("company", company)
        return "saved"
    
    @function_tool
    async def update_lead_email(self, context: RunContext, email: str):
        """Save the lead's email address
        
        Args:
            email: Email address
        """
        self.lead_tracker.update_field("email", email)
        return "saved"
    
    @function_tool
    async def update_lead_role(self, context: RunContext, role: str):
        """Save the lead's job title/role
        
        Args:
            role: Job title or role
        """
        self.lead_tracker.update_field("role", role)
        return "saved"
    
    @function_tool
    async def update_lead_use_case(self, context: RunContext, use_case: str):
        """Save what the lead wants to build or solve
        
        Args:
            use_case: Description of their use case or problem
        """
        self.lead_tracker.update_field("use_case", use_case)
        return "saved"
    
    @function_tool
    async def update_lead_team_size(self, context: RunContext, team_size: str):
        """Save the team size
        
        Args:
            team_size: Number of people on the team (e.g., "5-10", "just me", "50+")
        """
        self.lead_tracker.update_field("team_size", team_size)
        return "saved"
    
    @function_tool
    async def update_lead_timeline(self, context: RunContext, timeline: str):
        """Save when they're looking to implement
        
        Args:
            timeline: Implementation timeline (e.g., "next month", "Q2", "ASAP")
        """
        self.lead_tracker.update_field("timeline", timeline)
        return "saved"
    
    @function_tool
    async def end_call(self, context: RunContext):
        """End the call and provide a summary - use when customer says goodbye or indicates they're done
        
        No arguments needed - just call when the conversation is wrapping up
        """
        # Save the lead data
        filename = self.lead_tracker.save_to_file()
        
        # Generate summary
        lead = self.lead_tracker.data
        name = lead.get("name", "there")
        company = lead.get("company", "your company")
        use_case = lead.get("use_case", "your needs")
        timeline = lead.get("timeline", "soon")
        
        summary = f"""Thank you for your time, {name}! Let me quickly summarize:

- You're with {company}
- Looking to {use_case}
- Timeline: {timeline}

I've captured all your information and our team will reach out to {lead.get('email', 'you')} shortly to help you get started. 

Feel free to reach out anytime if you have more questions. Have a great day!"""
        
        logger.info(f"Call ended. Lead saved to: {filename}")
        return summary




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
    

