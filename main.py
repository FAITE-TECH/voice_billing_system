from speech_to_text import listen_to_speech
from gemini_api import extract_items
from pos_processor import generate_bill

def main():
    user_input = listen_to_speech()
    parsed = extract_items(user_input)
    bill = generate_bill(parsed)
    print("\n🧾 Final Bill:\n" + bill)

if __name__ == "__main__":
    main()
