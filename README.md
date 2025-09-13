import re
import json
import datetime
import random
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
# -----------------------------
# Expense Categorization (NLP)
# -----------------------------
class ExpenseCategorizer:
    categories = {
        "food": ["restaurant", "groceries", "snacks", "coffee"],
        "transport": ["uber", "bus", "train", "fuel"],
        "entertainment": ["movie", "netflix", "game", "concert"],

    
        "education": ["books", "tuition", "course", "stationery"],
        "utilities": ["electricity", "water", "internet", "phone"]
    }

    def categorize(self, description: str) -> str:
        description = description.lower()
        for category, keywords in self.categories.items():
            if any(re.search(rf"\b{kw}\b", description) for kw in keywords):
                return category
        return "other"

# -----------------------------
# Budget & Goal Management
# -----------------------------
class BudgetManager:
    def __init__(self, monthly_budget: float):
        self.monthly_budget = monthly_budget
        self.expenses: List[Dict] = []
        self.goals: Dict[str, float] = {}

    def add_expense(self, amount: float, description: str):
        categorizer = ExpenseCategorizer()
        category = categorizer.categorize(description)
        self.expenses.append({
            "amount": amount,
            "description": description,
            "category": category,
            "date": datetime.date.today().isoformat()
        })

    def set_goal(self, name: str, target_amount: float):
        self.goals[name] = target_amount

    def goal_progress(self) -> Dict[str, float]:
        progress = {}
        for goal, target in self.goals.items():
            spent = sum(e["amount"] for e in self.expenses if goal in e["description"].lower())
            progress[goal] = round((spent / target) * 100, 2)
        return progress

    def total_spent(self) -> float:
        return sum(e["amount"] for e in self.expenses)

    def check_budget_alert(self) -> str:
        spent = self.total_spent()
        if spent > self.monthly_budget:
            return "⚠️ Budget exceeded!"
        elif spent > 0.9 * self.monthly_budget:
            return "⚠️ Approaching budget limit!"
        return "✅ Within budget"

# -----------------------------
# Predictive Spending Analysis
# -----------------------------
class SpendingPredictor:
    def __init__(self, expense_history: List[Dict]):
        self.expense_history = expense_history

    def predict_next_month_spending(self) -> float:
        monthly_totals = {}
        for expense in self.expense_history:
            date = datetime.datetime.fromisoformat(expense["date"])
            key = f"{date.year}-{date.month}"
            monthly_totals.setdefault(key, 0)
            monthly_totals[key] += expense["amount"]

        months = sorted(monthly_totals.keys())
        X = np.array(range(len(months))).reshape(-1, 1)
        y = np.array([monthly_totals[m] for m in months])

        model = LinearRegression()
        model.fit(X, y)
        next_month_index = len(months)
        prediction = model.predict([[next_month_index]])
        return round(prediction[0], 2)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    manager = BudgetManager(monthly_budget=1000)

    # Add expenses
    manager.add_expense(120, "Groceries at supermarket")
    manager.add_expense(50, "Netflix subscription")
    manager.add_expense(30, "Bus fare")
    manager.add_expense(200, "Course enrollment")

    # Set goals
    manager.set_goal("course", 500)

    # Print insights
    print("Expenses:", json.dumps(manager.expenses, indent=2))
    print("Goal Progress:", manager.goal_progress())
    print("Budget Status:", manager.check_budget_alert())

    # Predict next month
    predictor = SpendingPredictor(manager.expenses)
    print("Predicted Next Month Spending:", predictor.predict_next_month_spending())
