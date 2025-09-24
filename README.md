# 🧾 Expense Tracker with Machine Learning

A desktop-based expense management application built with Python, Tkinter, SQLite, and Machine Learning.
This project goes beyond a simple tracker by integrating AI-powered features such as category suggestions, future expense prediction, and anomaly detection.

It is designed as a personal finance assistant that helps users record, analyze, and forecast expenses with ease.

## 🚀 Project Overview

Managing personal finances can be overwhelming, but this Expense Tracker with ML makes it simple and smart:

💰 Track daily expenses (date, category, description, amount)

🤖 Get ML-powered category suggestions based on past spending behavior

📈 Forecast future spending with machine learning regression models

🚨 Detect unusual patterns in your monthly expenses (anomalies)

📊 Visualize spending trends using plots and charts

With a user-friendly GUI and a robust backend, this app demonstrates the power of combining software engineering, data analytics, and machine learning.

## 🛠️ Skills & Technologies Showcased

Programming Languages: Python (OOP + scripting)

GUI Development: Tkinter (Treeview, Buttons, Forms, Dialogs)

Database Management: SQLite (CRUD operations, persistence)

Machine Learning Models:

Naive Bayes Classifier → Auto category suggestion

Linear Regression → Future expense prediction

Isolation Forest → Anomaly detection

Data Processing: Pandas & NumPy (grouping, aggregation, transformations)

Visualization: Matplotlib (spending history & prediction plots)

Model Serialization: Joblib (save/load trained models)

Software Engineering Practices:

Modular code (train_models.py, expense_tracker_gui.py, populate_sample_data.py)

Model retraining & versioning

Error handling & user prompts in GUI

.gitignore, requirements.txt for clean repo management

## ✨ Features

Expense Management

Add, view, and delete expenses

Organized storage in expenses.db (SQLite)

Category Suggestion (AI-powered)

Learns from past expense descriptions

Suggests most likely category automatically

Expense Prediction

Predicts next month’s spending using Linear Regression

Visualizes both historical expenses and predicted point

Anomaly Detection

Identifies months where spending is abnormally high/low

Useful for spotting overspending or unexpected costs

Sample Data Generator

Script (populate_sample_data.py) to pre-fill DB with realistic random expenses

Great for demo/testing without manual entry

Model Management

Retrain models anytime directly from the GUI

Models saved in /models for re-use

User-Friendly GUI

Tkinter-based interface with expense table

Buttons for all major operations:

Add Expense

Suggest Category

Predict Next Month

Detect Anomalies

Retrain Models

Delete Selected

## 🎯 Services This App Provides

✔️ Personal finance tracking
✔️ Automated categorization of expenses
✔️ Predictive analytics for future spending
✔️ Anomaly detection for financial health monitoring
✔️ Interactive data visualization
✔️ Easy-to-use GUI for non-technical users

This project highlights full-stack Python development with ML integration, making it a strong showcase of:

Data Science skills (ML, anomaly detection, regression)

Software Engineering (GUI apps, modular code, DB integration)

Problem-Solving (real-world application: finance management)
