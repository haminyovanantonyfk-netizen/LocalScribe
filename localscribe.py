import sounddevice as sd
import soundfile as sf
import ollama
from faster_whisper import WhisperModel
import datetime
import os

print("=" * 60)
print("🎙️  LOCALSCRIBE - Free Private Meeting Assistant")
print("=" * 60)
print("\nWhat would you like to do?")
print("1. Record a meeting (say 'stop' to end)")
print("2. Transcribe an existing audio file")
print()

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    print("\n🎤 Starting recording... Speak now!")
    print("Type 'stop' and press Enter when finished.")
    print("Recording in progress...\n")
    
    # Record until user says stop (max 5 minutes for now)
    recording = sd.rec(300 * 44100, samplerate=44100, channels=1, dtype='float32')
    input("Press Enter to stop recording...")
    sd.stop()
    
    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"meeting_{timestamp}.wav"
    sf.write(filename, recording, 44100)
    print(f"✅ Saved as {filename}")
    
    audio_file = filename
    
elif choice == "2":
    filename = input("Enter audio filename (without .wav): ").strip()
    audio_file = filename + ".wav"
    if not os.path.exists(audio_file):
        print(f"❌ File {audio_file} not found!")
        exit()
else:
    print("❌ Invalid choice!")
    exit()

# Step 2: Transcribe
print("\n🤖 Transcribing audio (this may take a few minutes)...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_file, beam_size=5, language="en")

full_transcript = " ".join([segment.text for segment in segments])

print("\n" + "=" * 60)
print("📝 FULL TRANSCRIPT:")
print("=" * 60)
print(full_transcript)
print("=" * 60)

# Step 3: Generate Summary
print("\n📋 Generating summary...")
summary_response = ollama.chat(model='phi3', messages=[
    {'role': 'user', 'content': f'''Summarize this meeting transcript in 3-4 bullet points:

{full_transcript}

Format:
- Key point 1
- Key point 2
- Key point 3'''},
])

summary = summary_response['message']['content']

# Step 4: Extract Action Items
print("✅ Extracting action items...")
action_response = ollama.chat(model='phi3', messages=[
    {'role': 'user', 'content': f'''Extract action items and tasks from this meeting transcript. List who is responsible if mentioned:

{full_transcript}

Format each action item as:
- [ ] Task description (Person responsible)

If no action items, say "No action items identified"'''},
])

action_items = action_response['message']['content']

# Step 5: Save Everything
output_file = f"meeting_notes_{timestamp}.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("MEETING NOTES - LocalScribe\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("📝 FULL TRANSCRIPT:\n")
    f.write("-" * 60 + "\n")
    f.write(full_transcript + "\n\n")
    
    f.write("📋 SUMMARY:\n")
    f.write("-" * 60 + "\n")
    f.write(summary + "\n\n")
    
    f.write("✅ ACTION ITEMS:\n")
    f.write("-" * 60 + "\n")
    f.write(action_items + "\n")

print("\n" + "=" * 60)
print("📋 SUMMARY:")
print("=" * 60)
print(summary)
print("=" * 60)

print("\n" + "=" * 60)
print("✅ ACTION ITEMS:")
print("=" * 60)
print(action_items)
print("=" * 60)

print(f"\n💾 All notes saved to: {output_file}")
print("\n🎉 Done! Your meeting has been processed!")