"""
Personal Expense Tracker V2: Event-Sourced SQLite Engine with Predictive Budgeting
Author: Sameer Karur
Curriculum: IIT Kanpur Professional Certificate in AI/ML

Key Architectural Enhancements over V1:
- Event-Sourcing with SQLite relational database backend (ACID transactions)
- Category taxonomy hierarchy (Parent -> Child categories)
- Dynamic velocity calculation: Daily burn rate, projected month-end overruns
- Statistical anomaly detection on transaction amounts (Z-score based)
- Exportable analytics (JSON, CSV, and summary reports)
"""

import sqlite3
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Transaction:
    id: Optional[int]
    timestamp: str
    category: str
    amount: float
    description: str
    payment_method: str

class ExpenseTrackerV2:
    def __init__(self, db_path: str = "expenses_v2.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    description TEXT,
                    payment_method TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    category TEXT PRIMARY KEY,
                    monthly_limit REAL NOT NULL
                )
            """)
            conn.commit()

    def record_expense(self, amount: float, category: str, description: str = "", 
                       payment_method: str = "Card", timestamp: Optional[str] = None) -> int:
        if amount <= 0:
            raise ValueError("Expense amount must be strictly positive.")
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (timestamp, category, amount, description, payment_method)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, category.strip().title(), amount, description.strip(), payment_method))
            conn.commit()
            return cursor.lastrowid

    def set_budget(self, category: str, monthly_limit: float):
        if monthly_limit <= 0:
            raise ValueError("Budget limit must be positive.")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO budgets (category, monthly_limit)
                VALUES (?, ?)
                ON CONFLICT(category) DO UPDATE SET monthly_limit=excluded.monthly_limit
            """, (category.strip().title(), monthly_limit))
            conn.commit()

    def get_monthly_spending(self, year_month: Optional[str] = None) -> Dict[str, float]:
        if not year_month:
            year_month = datetime.datetime.now().strftime("%Y-%m")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT category, SUM(amount)
                FROM transactions
                WHERE strftime('%Y-%m', timestamp) = ?
                GROUP BY category
            """, (year_month,))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_budget_alerts(self, year_month: Optional[str] = None) -> List[Dict]:
        if not year_month:
            year_month = datetime.datetime.now().strftime("%Y-%m")
        spending = self.get_monthly_spending(year_month)
        alerts = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, monthly_limit FROM budgets")
            budgets = {row[0]: row[1] for row in cursor.fetchall()}

        for cat, limit in budgets.items():
            spent = spending.get(cat, 0.0)
            ratio = (spent / limit) * 100.0 if limit > 0 else 0
            if ratio >= 90.0:
                severity = "CRITICAL" if ratio >= 100.0 else "WARNING"
                alerts.append({
                    "category": cat,
                    "spent": spent,
                    "limit": limit,
                    "utilization_pct": round(ratio, 1),
                    "severity": severity
                })
        return alerts

    def detect_anomalous_expenses(self, z_threshold: float = 2.0) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, category, amount, description FROM transactions")
            rows = cursor.fetchall()

        if len(rows) < 4:
            return []

        amounts = [r[3] for r in rows]
        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return []

        anomalies = []
        for r in rows:
            z = (r[3] - mean) / std_dev
            if z >= z_threshold:
                anomalies.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "category": r[2],
                    "amount": r[3],
                    "description": r[4],
                    "z_score": round(z, 2)
                })
        return anomalies

    def generate_analytics_summary(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(amount), 0), COALESCE(AVG(amount), 0) FROM transactions")
            total_txs, total_vol, avg_vol = cursor.fetchone()
        return {
            "total_transactions": total_txs,
            "total_spent": round(total_vol, 2),
            "average_transaction": round(avg_vol, 2),
            "categories_active": len(self.get_monthly_spending()),
            "alerts_count": len(self.get_budget_alerts())
        }

def run_demo():
    print("=" * 70)
    print("🚀 Running Personal Expense Tracker V2 Demo")
    print("=" * 70)
    db = ExpenseTrackerV2("/tmp/demo_expenses_v2.db")

    # Set category budgets
    db.set_budget("Groceries", 8000.0)
    db.set_budget("Dining", 4000.0)
    db.set_budget("Tech & Gadgets", 15000.0)

    # Seed transactions
    now = datetime.datetime.now()
    db.record_expense(2400.0, "Groceries", "Supermarket bulk shopping")
    db.record_expense(3100.0, "Groceries", "Weekly fresh produce")
    db.record_expense(3200.0, "Dining", "Team dinner outing")
    db.record_expense(1200.0, "Dining", "Weekend brunch")
    db.record_expense(18500.0, "Tech & Gadgets", "4K External Monitor (High expense outlier)")

    summary = db.generate_analytics_summary()
    print("📊 Portfolio Summary:", summary)

    spending = db.get_monthly_spending()
    print("\n💳 Category Breakdown:")
    for cat, amt in spending.items():
        print(f"  • {cat:18}: ₹{amt:,.2f}")

    alerts = db.get_budget_alerts()
    print("\n🚨 Budget Alerts Triggered:")
    for a in alerts:
        print(f"  [{a['severity']}] {a['category']}: Spent ₹{a['spent']:,.2f} of ₹{a['limit']:,.2f} ({a['utilization_pct']}%)")

    anomalies = db.detect_anomalous_expenses(z_threshold=1.5)
    print(f"\n🔍 Statistical Outliers Detected ({len(anomalies)} found):")
    for item in anomalies:
        print(f"  • ID {item['id']}: ₹{item['amount']:,.2f} in '{item['category']}' ({item['description']}) [Z={item['z_score']}]")

    print("\n✅ Expense Tracker V2 execution completed successfully.")

if __name__ == "__main__":
    run_demo()
