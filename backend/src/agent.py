import logging
import sqlite3
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

# Day 6 - Fraud Alert Database (SQLite)
FRAUD_DB_PATH = Path("shared-data/day6_fraud_cases.db")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(FRAUD_DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


class FraudAlertAgent(Agent):
    """Fraud Detection Representative for SecureBank - verifies suspicious transactions"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional Fraud Detection Representative for SafeGuard Bank. 
Your role is to contact customers about suspicious transactions and verify their legitimacy.

CONVERSATION FLOW:
1. Greet warmly: "Hello, this is the fraud department at SafeGuard Bank calling"
2. Ask for their name to look up their case
3. Once they provide name, call load_fraud_case tool
4. If case found: Ask security question from the database
5. If answer correct: Read transaction details and ask if they made it
6. If they confirm (yes): Call mark_transaction_safe tool
7. If they deny (no): Call mark_transaction_fraudulent tool
8. Thank them and end call

IMPORTANT RULES:
- Be calm, professional, and reassuring
- Never ask for card numbers, PINs, or passwords
- Use only the security question from the database
- Keep responses brief and clear
- If verification fails, politely end the call

BE CONVERSATIONAL AND NATURAL!""",
            tts=murf.TTS(
                voice="en-IN-priya",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8)
            )
        )
        self.fraud_case = None
        self.verification_passed = False
    
    async def on_enter(self) -> None:
        """Called when agent starts"""
        await self.session.generate_reply(
            instructions="Greet the customer warmly. Introduce yourself as a fraud detection representative from SafeGuard Bank. Ask for their name so you can look up their case."
        )
    
    @function_tool
    async def load_fraud_case(self, context: RunContext, user_name: str):
        """Load fraud case for a specific user from database
        
        Args:
            user_name: The customer's name to search for
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Search for matching case (case-insensitive)
        cursor.execute(
            "SELECT * FROM fraud_cases WHERE LOWER(userName) = LOWER(?)",
            (user_name,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.fraud_case = dict(row)
            logger.info(f"Fraud case loaded for {user_name}")
            return f"Case found. Security question: {self.fraud_case['securityQuestion']}"
        
        return f"No case found for {user_name}. Please check the name and try again."
    
    @function_tool
    async def verify_security_answer(self, context: RunContext, answer: str):
        """Verify the security answer provided by the customer
        
        Args:
            answer: The customer's answer to the security question
        """
        if not self.fraud_case:
            return "No case loaded. Please provide your name first."
        
        correct_answer = self.fraud_case.get("securityAnswer", "")
        
        if answer.lower().strip() == correct_answer.lower().strip():
            self.verification_passed = True
            transaction = f"""Verification successful! 

I need to verify a suspicious transaction on your card ending in {self.fraud_case.get('cardEnding')}:

Transaction: {self.fraud_case.get('transactionName')}
Amount: {self.fraud_case.get('transactionAmount')}
Time: {self.fraud_case.get('transactionTime')}
Location: {self.fraud_case.get('transactionLocation')}
Category: {self.fraud_case.get('transactionCategory')}

Did you make this transaction?"""
            return transaction
        else:
            return "I'm sorry, that answer doesn't match our records. For your security, I cannot proceed with this call. Please contact our fraud department directly. Goodbye."
    
    @function_tool
    async def mark_transaction_safe(self, context: RunContext):
        """Mark the transaction as safe/legitimate - customer confirmed they made it
        
        No arguments needed - marks the current loaded case as safe
        """
        if not self.fraud_case or not self.verification_passed:
            return "Cannot mark transaction - verification not completed."
        
        # Update case in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        outcome = f"Customer confirmed transaction as legitimate on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cursor.execute(
            "UPDATE fraud_cases SET case_status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP WHERE securityIdentifier = ?",
            ("confirmed_safe", outcome, self.fraud_case["securityIdentifier"])
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Transaction marked as SAFE for {self.fraud_case['userName']}")
        
        return """Thank you for confirming! Your transaction has been marked as safe and no further action is needed. 
Your card will continue to work normally. Have a great day!"""
    
    @function_tool
    async def mark_transaction_fraudulent(self, context: RunContext):
        """Mark the transaction as fraudulent - customer denies making it
        
        No arguments needed - marks the current loaded case as fraudulent
        """
        if not self.fraud_case or not self.verification_passed:
            return "Cannot mark transaction - verification not completed."
        
        # Update case in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        outcome = f"Customer denied transaction - fraud confirmed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Card blocked and dispute initiated."
        cursor.execute(
            "UPDATE fraud_cases SET case_status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP WHERE securityIdentifier = ?",
            ("confirmed_fraud", outcome, self.fraud_case["securityIdentifier"])
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Transaction marked as FRAUDULENT for {self.fraud_case['userName']}")
        
        return f"""Thank you for letting us know. I've immediately blocked your card ending in {self.fraud_case['cardEnding']} 
and initiated a fraud dispute for {self.fraud_case['transactionAmount']}. 

A new card will be mailed to you within 3-5 business days. You're protected from any fraudulent charges. 
Is there anything else I can help you with?"""



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
    
    # Start with Fraud Alert agent
    await session.start(
        agent=FraudAlertAgent(),
        room=ctx.room,
    )
    
    await ctx.connect()




if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    

