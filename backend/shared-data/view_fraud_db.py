"""
View Fraud Cases Database
Shows all fraud cases in a readable format
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "day6_fraud_cases.db"

def view_database():
    """Display all fraud cases from database"""
    
    if not DB_PATH.exists():
        print(f"❌ Database not found at: {DB_PATH}")
        print("Run setup_fraud_db.py first!")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM fraud_cases ORDER BY id")
    cases = cursor.fetchall()
    
    if not cases:
        print("No fraud cases in database")
        conn.close()
        return
    
    print(f"\n{'='*80}")
    print(f"FRAUD CASES DATABASE - {len(cases)} cases")
    print(f"{'='*80}\n")
    
    for case in cases:
        print(f"ID: {case['id']}")
        print(f"Customer: {case['userName']}")
        print(f"Security ID: {case['securityIdentifier']}")
        print(f"Card Ending: **** {case['cardEnding']}")
        print(f"Transaction: {case['transactionName']} - {case['transactionAmount']}")
        print(f"Time: {case['transactionTime']}")
        print(f"Location: {case['transactionLocation']}")
        print(f"Category: {case['transactionCategory']}")
        print(f"Source: {case['transactionSource']}")
        print(f"Status: {case['case_status']}")
        print(f"Outcome: {case['outcome'] or 'Pending'}")
        print(f"Security Question: {case['securityQuestion']}")
        print(f"Security Answer: {case['securityAnswer']}")
        print(f"Created: {case['created_at']}")
        print(f"Updated: {case['updated_at']}")
        print(f"{'-'*80}\n")
    
    conn.close()

if __name__ == "__main__":
    view_database()
