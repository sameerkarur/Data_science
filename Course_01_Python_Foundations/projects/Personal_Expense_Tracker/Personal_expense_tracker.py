import csv
import os
from datetime import datetime

expenses = []
monthly_budget = 0.0


def load_expenses(filename="expenses.csv"):
    global expenses
    if not os.path.exists(filename):
        return
    with open(filename, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 4:
                date, category, amount, description = row
                expenses.append({
                    "date": date,
                    "category": category,
                    "amount": float(amount),
                    "description": description
                })


def save_expenses(filename="expenses.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for exp in expenses:
            writer.writerow([exp["date"], exp["category"], exp["amount"], exp["description"]])
    print("Expenses saved successfully.")


def add_expense():
    date = datetime.now().strftime("%Y-%m-%d")
    category = input("Enter category (Food, Transport, etc.): ")
    amount = float(input("Enter amount spent: "))
    description = input("Enter a short description: ")

    expenses.append({
        "date": date,
        "category": category,
        "amount": amount,
        "description": description
    })
    print("Expense added.\n")


def view_expenses():
    if not expenses:
        print("No expenses found.\n")
        return
    print("\n----- Expense List -----")
    for e in expenses:
        print(f"{e['date']} | {e['category']} | ${e['amount']} | {e['description']}")
    print("------------------------\n")


def set_and_track_budget():
    global monthly_budget
    if monthly_budget == 0:
        monthly_budget = float(input("Enter your monthly budget: "))
        print(f"Budget for the month set to ${monthly_budget}\n")

    total = sum(e["amount"] for e in expenses)

    if total > monthly_budget:
        print("WARNING: You have exceeded your budget!")
        print(f"Total spent: ${total:.2f} | Budget: ${monthly_budget:.2f}\n")
    else:
        remaining = monthly_budget - total
        print(f"Total spent: ${total:.2f}")
        print(f"You have ${remaining:.2f} remaining for the month.\n")


def menu():
    load_expenses()
    while True:
        print("========== Expense Tracker ==========")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Track Budget")
        print("4. Save Expenses")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            set_and_track_budget()
        elif choice == "4":
            save_expenses()
        elif choice == "5":
            save_expenses()
            print("Goodbye!")
            break
        else:
            print("Invalid selection, try again.\n")


if __name__ == "__main__":
    menu()
