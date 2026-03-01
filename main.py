import os
from faster_whisper import WhisperModel

# Forces the script to see your ffmpeg.exe
os.environ["PATH"] += os.pathsep + os.getcwd()

def run_agent():
    print("--- AURALOG_AGENT: MP3 TRANSCRIPT MODE ---")
    
    # List only MP3 files for a cleaner view
    print("\nAvailable MP3 files:")
    mp3_files = [f for f in os.listdir() if f.lower().endswith('.mp3')]
    
    if not mp3_files:
        print("No MP3 files found in the folder!")
        return

    for f in mp3_files:
        print(f" -> {f}")

    # ASK ME: Input prompt
    file_to_process = input("\nEnter the MP3 filename: ").strip()

    if not os.path.exists(file_to_process):
        print(f"Error: Could not find '{file_to_process}'.")
        return

    try:
        print("\nStep 1: Loading Intelligence...")
        # int8 keeps the 1.44 GiB memory error away
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        print(f"Step 2: Transcribing {file_to_process}...")
        segments, info = model.transcribe(file_to_process, beam_size=5, vad_filter=True)
        
        print("\n--- TRANSCRIPT START ---")
        
        # Save to a clean text transcript
        with open("transcript.txt", "w", encoding="utf-8") as f:
            for segment in segments:
                # Simple timestamp and text
                output = f"[{segment.start:.2f}s] {segment.text}"
                print(output)
                f.write(output + "\n")
        
        print(f"\nSUCCESS: Transcript saved to 'transcript.txt'")
            
    except Exception as e:
        print(f"System Check Failed: {e}")

if __name__ == "__main__":
    run_agent()