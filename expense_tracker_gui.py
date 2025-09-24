import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3, os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from train_models import train_all  # Import the specific function we need

DB_PATH = "expenses.db"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Build GUI ----
root = tk.Tk()
root.title("Expense Tracker with ML")

# ---- Database setup ----
conn = sqlite3.connect(DB_PATH)
db_cursor = conn.cursor()
db_cursor.execute('''CREATE TABLE IF NOT EXISTS expenses
             (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, category TEXT, description TEXT, amount REAL)''')
conn.commit()

# ---- Helper: load models if present ----
def load_models():
    models = {}
    cat_path = os.path.join(MODELS_DIR, "cat_model.pkl")
    pred_path = os.path.join(MODELS_DIR, "pred_model.pkl")
    iso_path = os.path.join(MODELS_DIR, "iso_model.pkl")

    if os.path.exists(cat_path):
        try:
            models['cat'] = joblib.load(cat_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load category model: {e}")
            models['cat'] = None
    else:
        models['cat'] = None

    if os.path.exists(pred_path):
        try:
            models['pred_meta'] = joblib.load(pred_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load prediction model: {e}")
            models['pred_meta'] = None
    else:
        models['pred_meta'] = None

    if os.path.exists(iso_path):
        try:
            models['iso'] = joblib.load(iso_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load anomaly model: {e}")
            models['iso'] = None
    else:
        models['iso'] = None

    return models

MODELS = load_models()

# ---- Functions ----
def add_expense():
    date = date_entry.get().strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    category = category_var.get().strip()
    description = desc_entry.get().strip()
    try:
        amount = float(amount_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Amount must be a number")
        return
    try:
        db_cursor.execute("INSERT INTO expenses (date, category, description, amount) VALUES (?, ?, ?, ?)",
                  (date, category, description, amount))
        conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to add expense: {e}")
        return
    load_expenses()
    clear_entries()

def suggest_category():
    desc = desc_entry.get().strip()
    if not desc:
        messagebox.showwarning("Enter description", "Type description to get suggestion.")
        return
    model = MODELS.get('cat')
    if model is None:
        messagebox.showinfo("Model missing", "Category model not found. Retrain models first.")
        return
    pred = model.predict([desc])[0]
    category_var.set(pred)
    messagebox.showinfo("Suggested", f"Suggested Category: {pred}")

def load_expenses():
    for row in tree.get_children():
        tree.delete(row)
    try:
        for row in db_cursor.execute("SELECT * FROM expenses ORDER BY date DESC").fetchall():
            tree.insert("", "end", values=row)
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to load expenses: {e}")

def delete_expense():
    sel = tree.selection()
    if not sel:
        messagebox.showwarning("Select", "Select a row to delete.")
        return
    try:
        for item in sel:
            row = tree.item(item)['values']
            exp_id = row[0]
            db_cursor.execute("DELETE FROM expenses WHERE id=?", (exp_id,))
        conn.commit()
        load_expenses()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to delete expense: {e}")

def predict_next_month():
    meta = MODELS.get('pred_meta')
    if meta is None:
        messagebox.showinfo("Model missing", "Prediction model not found. Retrain models first.")
        return
    # meta may be fallback ("mean", value) or dict with model
    if isinstance(meta, tuple) and meta[0] == "mean":
        pred = meta[1]
        messagebox.showinfo("Prediction", f"Predicted next month expense (fallback mean): ₹{pred:.2f}")
        return
    model = meta['model']
    last_idx = meta['last_idx']
    next_idx = last_idx + 1
    pred_val = model.predict([[next_idx]])[0]
    # show simple plot of history + prediction
    dates = meta.get('monthly_dates', [])
    vals = meta.get('monthly_values', [])
    plt.figure()
    plt.plot(dates, vals, marker='o', label='history')
    plt.xticks(rotation=45)
    plt.title("Monthly expenses (history + predicted point)")
    # show predicted point with next-month label
    next_label = (datetime.strptime(dates[-1]+"-01", "%Y-%m-%d") + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    plt.scatter([next_label], [pred_val], color='red', label=f'predicted: ₹{pred_val:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    messagebox.showinfo("Prediction", f"Predicted next month expense: ₹{pred_val:.2f}")

def detect_anomalies():
    iso_meta = MODELS.get('iso')
    if iso_meta is None:
        messagebox.showinfo("Model missing", "Anomaly model not found. Retrain models first.")
        return
    model = iso_meta['model']
    dates = iso_meta['dates']
    vals = iso_meta['values']
    import numpy as np
    X = np.array(vals).reshape(-1,1)
    preds = model.predict(X)  # 1 = normal, -1 = anomaly
    anomalies = [(d,v) for d,v,p in zip(dates, vals, preds) if p == -1]
    if not anomalies:
        messagebox.showinfo("Anomalies", "No anomalous months detected.")
    else:
        s = "\n".join([f"{d}: ₹{v:.2f}" for d,v in anomalies])
        messagebox.showwarning("Anomalies detected", s)

def retrain_models():
    # call training function in this process (train_models.train_all)
    if messagebox.askyesno("Retrain", "Retraining will read DB and overwrite models. Continue?"):
        train_all()  # uses same folder
        # reload models
        global MODELS
        MODELS = load_models()
        messagebox.showinfo("Retrain", "Retraining finished and models reloaded.")

def on_closing():
    try:
        conn.close()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

def clear_entries():
    date_entry.delete(0, tk.END)
    category_var.set("")
    desc_entry.delete(0, tk.END)
    amount_entry.delete(0, tk.END)

# Input frame
# Removed duplicate GUI build code to avoid redundancy

# Input frame
frm = ttk.Frame(root, padding=8)
frm.grid(row=0, column=0, sticky="ew")

ttk.Label(frm, text="Date (YYYY-MM-DD)").grid(row=0, column=0)
date_entry = ttk.Entry(frm, width=14)
date_entry.grid(row=0, column=1)
date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

ttk.Label(frm, text="Category").grid(row=0, column=2)
category_var = tk.StringVar()
category_box = ttk.Combobox(frm, textvariable=category_var, values=["Food","Travel","Shopping","Bills","Other"], width=18)
category_box.grid(row=0, column=3)

ttk.Label(frm, text="Description").grid(row=1, column=0)
desc_entry = ttk.Entry(frm, width=40)
desc_entry.grid(row=1, column=1, columnspan=3, sticky="ew")

ttk.Label(frm, text="Amount").grid(row=0, column=4)
amount_entry = ttk.Entry(frm, width=12)
amount_entry.grid(row=0, column=5)

ttk.Button(frm, text="Add Expense", command=add_expense).grid(row=2, column=1, pady=6)
ttk.Button(frm, text="Suggest Category", command=suggest_category).grid(row=2, column=2)
ttk.Button(frm, text="Predict Next Month", command=predict_next_month).grid(row=2, column=3)
ttk.Button(frm, text="Detect Anomalies", command=detect_anomalies).grid(row=2, column=4)
ttk.Button(frm, text="Retrain Models", command=retrain_models).grid(row=2, column=5)

# Treeview
cols = ("ID","date","category","description","amount")
tree = ttk.Treeview(root, columns=cols, show="headings", height=12)
for c in cols:
    tree.heading(c, text=c)
    tree.column(c, width=120)
tree.grid(row=3, column=0, padx=8, pady=8)

ttk.Button(root, text="Delete Selected", command=delete_expense).grid(row=4, column=0, pady=6)

load_expenses()
root.mainloop()
