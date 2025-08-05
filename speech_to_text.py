import speech_recognition as sr

def listen_to_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎙️ Speak your billing command...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
