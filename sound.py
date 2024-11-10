import pyttsx3

engine = pyttsx3.init()
text = "This if 50 rupee note."
engine.say(text)

# Customize speech rate and volume (optional)
# engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

engine.runAndWait()