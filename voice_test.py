import sounddevice as sd
import soundfile as sf
import ollama
from faster_whisper import WhisperModel

print("🎤 Voice Recorder - Say something!")
print("Recording for 5 seconds...")

# Record audio from microphone
try:
    # Record for 5 seconds
    recording = sd.rec(5 * 44100, samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    
    # Save the recording
    sf.write('temp_audio.wav', recording, 44100)
    print("✅ Audio recorded!")
    
    # Now transcribe it with Whisper
    print("🤖 Transcribing audio... (this may take a moment)")
    
    # Load the Whisper model (small is fast, medium is more accurate)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    # Transcribe the audio
    segments, info = model.transcribe("temp_audio.wav", beam_size=5)
    
    # Get the full text
    transcription = " ".join([segment.text for segment in segments])
    
    print("\n📝 YOU SAID:")
    print("=" * 50)
    print(transcription)
    print("=" * 50)
    
    # Now ask the AI to respond
    print("\n💬 Asking AI to respond...")
    response = ollama.chat(model='phi3', messages=[
        {'role': 'user', 'content': f'You heard this: "{transcription}". Respond helpfully and briefly.'},
    ])
    
    print("\n🤖 AI RESPONSE:")
    print("=" * 50)
    print(response['message']['content'])
    print("=" * 50)
    
except KeyboardInterrupt:
    print("\n⏹️  Recording stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")