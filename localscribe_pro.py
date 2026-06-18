import sounddevice as sd
import soundfile as sf
import ollama
from faster_whisper import WhisperModel
import datetime
import os
import json
import sys
from pathlib import Path

# Try to import cloud AI libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except:
    GOOGLE_AVAILABLE = False

print("=" * 70)
print("🎙️  LOCALSCRIBE PRO - The Ultimate Meeting Assistant")
print("=" * 70)
print("\n🌟 Features:")
print("  ✓ Real-time transcription")
print("  ✓ System audio recording (Zoom/Teams)")
print("  ✓ Cloud AI APIs (OpenAI, Anthropic, Google)")
print("  ✓ Local AI (Ollama) - 100% offline")
print("=" * 70)

# Configuration file for API keys
CONFIG_FILE = "config.json"

def load_config():
    """Load API keys from config file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    """Save API keys to config file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def setup_api_keys():
    """Setup API keys interactively"""
    config = load_config()
    
    print("\n🔑 API Key Setup (press Enter to skip)")
    
    if not config.get('openai_key'):
        key = input("OpenAI API Key: ").strip()
        if key:
            config['openai_key'] = key
    
    if not config.get('anthropic_key'):
        key = input("Anthropic API Key: ").strip()
        if key:
            config['anthropic_key'] = key
    
    if not config.get('google_key'):
        key = input("Google Gemini API Key: ").strip()
        if key:
            config['google_key'] = key
    
    save_config(config)
    return config

def get_ai_client(config):
    """Get the AI client based on user choice"""
    print("\n🤖 Select AI Provider:")
    print("1. Ollama (Local, Free, Offline)")
    if OPENAI_AVAILABLE and config.get('openai_key'):
        print("2. OpenAI GPT (Cloud, Fast, Accurate)")
    if ANTHROPIC_AVAILABLE and config.get('anthropic_key'):
        print("3. Anthropic Claude (Cloud, Fast, Accurate)")
    if GOOGLE_AVAILABLE and config.get('google_key'):
        print("4. Google Gemini (Cloud, Fast, Accurate)")
    
    choice = input("\nSelect (1-4, default=1): ").strip() or "1"
    
    if choice == "1":
        return "ollama", None
    elif choice == "2" and config.get('openai_key'):
        return "openai", OpenAI(api_key=config['openai_key'])
    elif choice == "3" and config.get('anthropic_key'):
        return "anthropic", Anthropic(api_key=config['anthropic_key'])
    elif choice == "4" and config.get('google_key'):
        genai.configure(api_key=config['google_key'])
        return "google", genai.GenerativeModel('gemini-pro')
    else:
        return "ollama", None

def transcribe_audio(audio_file, use_gpu=False):
    """Transcribe audio using faster-whisper"""
    print("\n🤖 Transcribing audio...")
    
    # Use tiny model for speed, small for better accuracy
    model_size = "tiny" if not use_gpu else "small"
    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "int8"
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_file, beam_size=5, language="en")
        
        transcription = " ".join([segment.text for segment in segments])
        return transcription
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def generate_summary(transcription, ai_provider, ai_client):
    """Generate summary using selected AI provider"""
    print("\n📋 Generating summary...")
    
    prompt = f"""Summarize this meeting transcript in 3-4 bullet points:

{transcription}

Format:
- Key point 1
- Key point 2
- Key point 3"""

    try:
        if ai_provider == "ollama":
            response = ollama.chat(model='phi3', messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        
        elif ai_provider == "openai":
            response = ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        
        elif ai_provider == "anthropic":
            response = ai_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif ai_provider == "google":
            response = ai_client.generate_content(prompt)
            return response.text
    
    except Exception as e:
        print(f"❌ Summary generation error: {e}")
        return "Error generating summary"

def extract_action_items(transcription, ai_provider, ai_client):
    """Extract action items using selected AI provider"""
    print("✅ Extracting action items...")
    
    prompt = f"""Extract action items and tasks from this meeting transcript:

{transcription}

Format each action item as:
- [ ] Task description (Person responsible)

If no action items, say "No action items identified" """

    try:
        if ai_provider == "ollama":
            response = ollama.chat(model='phi3', messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        
        elif ai_provider == "openai":
            response = ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        
        elif ai_provider == "anthropic":
            response = ai_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif ai_provider == "google":
            response = ai_client.generate_content(prompt)
            return response.text
    
    except Exception as e:
        print(f"❌ Action items extraction error: {e}")
        return "Error extracting action items"

def record_microphone(duration=300):
    """Record from microphone"""
    print("\n🎤 Recording from microphone...")
    print(f"Recording for up to {duration} seconds. Press Ctrl+C to stop early.")
    
    try:
        recording = sd.rec(duration * 44100, samplerate=44100, channels=1, dtype='float32')
        input("Press Enter to stop recording...")
        sd.stop()
        return recording
    except KeyboardInterrupt:
        sd.stop()
        print("\n⏹️ Recording stopped")
        return recording

def record_system_audio(duration=300):
    """Record system audio (for Zoom/Teams)"""
    print("\n🎧 SYSTEM AUDIO RECORDING")
    print("⚠️  NOTE: This requires VB-Cable or similar virtual audio cable")
    print("📥 Download: https://vb-audio.com/Cable/")
    print("\nAfter installing VB-Cable:")
    print("1. Set VB-Cable as your default recording device")
    print("2. Set VB-Cable as the output device for Zoom/Teams")
    print("\nStarting recording in 5 seconds...")
    
    import time
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("🔴 Recording system audio...")
    try:
        recording = sd.rec(duration * 44100, samplerate=44100, channels=1, dtype='float32')
        input("Press Enter to stop recording...")
        sd.stop()
        return recording
    except KeyboardInterrupt:
        sd.stop()
        print("\n⏹️ Recording stopped")
        return recording

def real_time_transcription(duration=60):
    """Real-time transcription as you speak"""
    print("\n🎤 REAL-TIME TRANSCRIPTION MODE")
    print("Speak now! I'll transcribe as you go.")
    print("Press Ctrl+C to stop.\n")
    
    # Load Whisper model
    print("Loading transcription model...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Ready! Start speaking...\n")
    
    try:
        # Stream audio in chunks
        with sd.InputStream(samplerate=44100, channels=1, dtype='float32', 
                           callback=lambda indata, frames, time, status: None) as stream:
            
            chunk_duration = 5  # seconds
            total_recorded = 0
            
            while total_recorded < duration:
                # Record a chunk
                chunk = sd.rec(int(chunk_duration * 44100), samplerate=44100, 
                              channels=1, dtype='float32')
                sd.wait()
                
                # Save temporarily
                sf.write('temp_chunk.wav', chunk, 44100)
                
                # Transcribe
                segments, _ = model.transcribe('temp_chunk.wav', language="en")
                text = " ".join([segment.text for segment in segments])
                
                if text.strip():
                    print(f"📝 {text}")
                
                total_recorded += chunk_duration
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopped")
    finally:
        # Clean up
        if os.path.exists('temp_chunk.wav'):
            os.remove('temp_chunk.wav')

def main():
    # Setup
    config = setup_api_keys()
    ai_provider, ai_client = get_ai_client(config)
    
    print(f"\n✅ Using: {ai_provider.upper()}")
    
    # Main menu
    print("\n" + "=" * 70)
    print("📋 MAIN MENU")
    print("=" * 70)
    print("1. 🎤 Record meeting (microphone)")
    print("2. 🎧 Record system audio (Zoom/Teams)")
    print("3. 🎬 Real-time transcription")
    print("4. 📁 Transcribe existing audio file")
    print("5. ⚙️  Change AI provider")
    print("6. ❌ Exit")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "6":
        print("👋 Goodbye!")
        return
    
    if choice == "5":
        ai_provider, ai_client = get_ai_client(config)
        # Restart menu
        choice = input("\nSelect option (1-4): ").strip()
    
    # Handle recording
    if choice == "1":
        recording = record_microphone()
    elif choice == "2":
        recording = record_system_audio()
    elif choice == "3":
        real_time_transcription()
        return
    elif choice == "4":
        filename = input("Enter audio filename (with .wav): ").strip()
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        audio_file = filename
    else:
        print("❌ Invalid choice")
        return
    
    # Save recording
    if choice in ["1", "2"]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = f"meeting_{timestamp}.wav"
        sf.write(audio_file, recording, 44100)
        print(f"✅ Saved as {audio_file}")
    
    # Transcribe
    transcription = transcribe_audio(audio_file)
    
    if not transcription:
        print("❌ No transcription generated")
        return
    
    print("\n" + "=" * 70)
    print("📝 FULL TRANSCRIPT:")
    print("=" * 70)
    print(transcription)
    print("=" * 70)
    
    # Generate summary and action items
    summary = generate_summary(transcription, ai_provider, ai_client)
    action_items = extract_action_items(transcription, ai_provider, ai_client)
    
    # Save to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"meeting_notes_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MEETING NOTES - LocalScribe PRO\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AI Provider: {ai_provider}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📝 FULL TRANSCRIPT:\n")
        f.write("-" * 70 + "\n")
        f.write(transcription + "\n\n")
        
        f.write("📋 SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(summary + "\n\n")
        
        f.write("✅ ACTION ITEMS:\n")
        f.write("-" * 70 + "\n")
        f.write(action_items + "\n")
    
    # Display results
    print("\n" + "=" * 70)
    print("📋 SUMMARY:")
    print("=" * 70)
    print(summary)
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ ACTION ITEMS:")
    print("=" * 70)
    print(action_items)
    print("=" * 70)
    
    print(f"\n💾 All notes saved to: {output_file}")
    print(f"🎵 Audio saved to: {audio_file}")
    print("\n🎉 Done! Your meeting has been processed!")

if __name__ == "__main__":
    main()