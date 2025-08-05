import json
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def clean_json_block(text):
    """Remove markdown code blocks and whitespace around JSON."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()

def preprocess_text(text):
    """Remove common filler phrases and lowercase the input."""
    fillers = [
        "in one", "please", "i want", "can i have", "could you get me",
        "would you get me", "and then", "also", "may i have", "give me",
        "for me", "just"
    ]
    text = text.lower()
    for filler in fillers:
        text = text.replace(filler, "")
    # Remove extra spaces and commas at ends
    text = re.sub(r"[ ,]+$", "", text)
    text = re.sub(r"^[ ,]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fallback_parse(text):
    """
    Basic rule-based parser if LLM fails.
    Converts number words and extracts items separated by commas/and.
    """
    number_words = {
        "zero":0, "one":1, "two":2, "three":3, "four":4,
        "five":5, "six":6, "seven":7, "eight":8, "nine":9,
        "ten":10
    }

    # Clean fillers again just in case
    text = preprocess_text(text)

    # Split by commas and "and"
    parts = re.split(r",| and ", text)

    items = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        # Assume quantity first token if number word or digit
        qty = 1
        item_tokens = []
        if tokens:
            first = tokens[0]
            if first.isdigit():
                qty = int(first)
                item_tokens = tokens[1:]
            elif first in number_words:
                qty = number_words[first]
                item_tokens = tokens[1:]
            else:
                item_tokens = tokens
        item_name = " ".join(item_tokens).strip()
        if item_name:
            items.append({"item": item_name, "quantity": qty})

    return items

def extract_items(user_input, retry=True):
    user_input_clean = preprocess_text(user_input)
    prompt = f"""
You are an assistant that extracts items and quantities from billing voice commands.

### Task:
Convert the following spoken billing command into a list of item objects in valid JSON format.

### Important:
- Convert number words ("one", "two") to digits.
- Ignore filler words such as "in one", "please", "I want".
- Separate multiple items connected by "and", commas, or spaces.
- If quantity is missing, assume 1.

### Examples:

Input: "two black rice and one plum"
Output:
[
  {{ "item": "black rice", "quantity": 2 }},
  {{ "item": "plum", "quantity": 1 }}
]

Input: "one arabica coffee, three tea bags"
Output:
[
  {{ "item": "arabica coffee", "quantity": 1 }},
  {{ "item": "tea bags", "quantity": 3 }}
]

Input: "two black rice in one"
Output:
[
  {{ "item": "black rice", "quantity": 2 }}
]

### Input:
"{user_input_clean}"

### Output:
"""

    response = model.generate_content(prompt)
    try:
        raw = clean_json_block(response.text)
        items = json.loads(raw)
        if isinstance(items, list):
            return items
        else:
            print("❌ Gemini response JSON is not a list, using fallback parser.")
            return fallback_parse(user_input_clean)
    except json.JSONDecodeError:
        print("❌ Failed to parse Gemini response JSON, falling back to rule-based parser.")
        print("Gemini response was:\n", response.text)
        if retry:
            return fallback_parse(user_input_clean)
        return []

# Example usage
if __name__ == "__main__":
    while True:
        command = input("🎙️ Speak your billing command (or type 'exit' to quit):\n> ")
        if command.strip().lower() == "exit":
            break
        items = extract_items(command)
        if not items:
            print("No items found. Please try again.")
            continue

        print("\n🧾 Parsed Bill Items:")
        total_amount = 0
        # Example prices (replace with your pricing logic)
        price_list = {
            "black rice": 6.0,
            "plum": 15.0,
            "arabica coffee": 10.0,
            "chips": 30.0,
            "cola": 40.0,
        }
        for i in items:
            item = i.get("item", "").lower()
            qty = i.get("quantity", 1)
            price = price_list.get(item, 0)
            amount = price * qty
            total_amount += amount
            print(f"{qty} x {item.title()} = ₹{amount:.2f}")
        print(f"\nTotal: ₹{total_amount:.2f}\n")
