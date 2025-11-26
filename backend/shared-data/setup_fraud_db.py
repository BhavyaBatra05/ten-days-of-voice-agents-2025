"""
Setup script for Day 6 Fraud Alert Database
Creates SQLite database with sample fraud cases
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "day6_fraud_cases.db"

def setup_database():
    """Create and populate fraud cases database"""
    
    # Remove existing database if present
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    # Create connection
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create fraud_cases table
    cursor.execute("""
        CREATE TABLE fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT NOT NULL,
            securityIdentifier TEXT UNIQUE NOT NULL,
            securityQuestion TEXT NOT NULL,
            securityAnswer TEXT NOT NULL,
            cardEnding TEXT NOT NULL,
            case_status TEXT DEFAULT 'pending_review',
            transactionAmount TEXT NOT NULL,
            transactionName TEXT NOT NULL,
            transactionTime TEXT NOT NULL,
            transactionCategory TEXT NOT NULL,
            transactionSource TEXT NOT NULL,
            transactionLocation TEXT NOT NULL,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert sample fraud cases
    fraud_cases = [
        (
            "Amit Patel",
            "12345",
            "What city were you born in?",
            "Delhi",
            "4242",
            "pending_review",
            "₹1,04,562",
            "TechBazar Online",
            "November 26, 2025 at 11:45 PM",
            "e-commerce",
            "techbazar.in",
            "Bengaluru, Karnataka",
            None
        ),
        (
            "Priya Sharma",
            "67890",
            "What city were you born in?",
            "Mumbai",
            "8888",
            "pending_review",
            "₹3,02,100",
            "Luxury Jewels India",
            "November 27, 2025 at 2:30 AM",
            "jewelry",
            "luxuryjewels.co.in",
            "Delhi, NCR",
            None
        ),
        (
            "Rahul Verma",
            "24680",
            "What is your favorite color?",
            "Blue",
            "1234",
            "pending_review",
            "₹7,560",
            "GameZone India",
            "November 26, 2025 at 9:15 PM",
            "gaming",
            "gamezone.in",
            "Pune, Maharashtra",
            None
        ),
        (
            "Neha Singh",
            "13579",
            "What was your first pet's name?",
            "Sheru",
            "5678",
            "pending_review",
            "₹1,80,500",
            "ElectroMart",
            "November 27, 2025 at 4:20 AM",
            "electronics",
            "electromart.com",
            "Hyderabad, Telangana",
            None
        ),
        (
            "Rajesh Kumar",
            "97531",
            "What is your favorite food?",
            "Biryani",
            "9999",
            "pending_review",
            "₹47,680",
            "Fashion Hub India",
            "November 26, 2025 at 7:45 PM",
            "clothing",
            "fashionhub.in",
            "Chennai, Tamil Nadu",
            None
        )
    ]
    
    cursor.executemany("""
        INSERT INTO fraud_cases (
            userName, securityIdentifier, securityQuestion, securityAnswer,
            cardEnding, case_status, transactionAmount, transactionName,
            transactionTime, transactionCategory, transactionSource,
            transactionLocation, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, fraud_cases)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully at: {DB_PATH}")
    print(f"✅ Added {len(fraud_cases)} fraud cases")
    
    # Verify
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT userName, transactionAmount, case_status FROM fraud_cases")
    rows = cursor.fetchall()
    print("\n📋 Fraud Cases:")
    for row in rows:
        print(f"  - {row[0]}: {row[1]} ({row[2]})")
    conn.close()

if __name__ == "__main__":
    setup_database()
