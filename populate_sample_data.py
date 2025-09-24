import sqlite3
from datetime import datetime, timedelta
import random

DB_PATH = "expenses.db"

def generate_sample_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    categories = ["Food", "Travel", "Shopping", "Bills", "Other"]
    descriptions = {
        "Food": ["Lunch at cafe", "Groceries", "Dinner at restaurant"],
        "Travel": ["Uber ride", "Bus ticket", "Train fare"],
        "Shopping": ["Clothes", "Electronics", "Books"],
        "Bills": ["Electricity bill", "Water bill", "Internet bill"],
        "Other": ["Gift", "Donation", "Miscellaneous"]
    }

    start_date = datetime.now() - timedelta(days=180)  # 6 months ago
    for i in range(50):  # 50 sample entries
        date = start_date + timedelta(days=random.randint(0, 180))
        category = random.choice(categories)
        description = random.choice(descriptions[category])
        amount = round(random.uniform(5.0, 100.0), 2)
        c.execute("INSERT INTO expenses (date, category, description, amount) VALUES (?, ?, ?, ?)",
                  (date.strftime("%Y-%m-%d"), category, description, amount))

    conn.commit()
    conn.close()
    print("Sample data generated and inserted into the database.")

if __name__ == "__main__":
    generate_sample_data()
