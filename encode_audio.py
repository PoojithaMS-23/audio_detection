import base64

# Path to your MP3 file (update if your file is somewhere else)
mp3_file = r"C:\Users\rajal\Downloads\sample voice 1.mp3"

# Read the MP3 in binary mode
with open(mp3_file, "rb") as f:
    audio_bytes = f.read()

# Encode to Base64 and convert to string
audio_base64 = base64.b64encode(audio_bytes).decode()

# Save the Base64 into a clean text file
with open("audio_clean.txt", "w") as f:
    f.write(audio_base64)

print("✅ Base64 ready in audio_clean.txt")
