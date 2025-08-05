import csv

def load_inventory_from_csv(file_path="inventory.csv"):
    inventory = {}
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_name = row["Product_Name"].strip().lower()
            try:
                inventory[product_name] = float(row["Unit_Price"])
            except ValueError:
                inventory[product_name] = 0.0
    return inventory

def generate_bill(parsed_items):
    inventory = load_inventory_from_csv()
    total = 0
    bill_lines = []

    for entry in parsed_items:
        item_name = entry["item"].lower()
        quantity = entry["quantity"]
        price = inventory.get(item_name)

        if price is not None:
            cost = price * quantity
            total += cost
            bill_lines.append(f"{quantity} x {item_name.title()} = ₹{cost:.2f}")
        else:
            bill_lines.append(f"{item_name.title()} not found in inventory")

    bill = "\n".join(bill_lines)
    bill += f"\n\nTotal: ₹{total:.2f}"
    return bill
