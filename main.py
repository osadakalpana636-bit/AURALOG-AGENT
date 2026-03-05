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
        print(f" -> {{f}}")

    # ASK ME: Input prompt
    file_to_process = input("\nEnter the MP3 filename: ").strip()

    # Validate file extension
    if not file_to_process.lower().endswith('.mp3'):
        print("Error: Please enter a valid MP3 filename (must end with .mp3).")
        return

    # Validate file exists
    if not os.path.exists(file_to_process):
        print(f"Error: Could not find '{{file_to_process}}'.")
        return

    try:
        print("\nStep 1: Loading Intelligence...")
        # int8 keeps the 1.44 GiB memory error away
        model = WhisperModel("base", device="cpu", compute_type="int8")

        print(f"Step 2: Transcribing {{file_to_process}}...")
        segments, info = model.transcribe(file_to_process, beam_size=5, vad_filter=True)
        
        # Convert generator to list for safer processing
        segments = list(segments)
        
        if not segments:
            print("Warning: No speech detected in the audio file.")
            return

        print("\n--- TRANSCRIPT START ---")

        # Generate output filename based on input file
        base_name = os.path.splitext(file_to_process)[0]
        output_file = f"{{base_name}}_transcript.txt"

        # Save to a clean text transcript
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in segments:
                # Simple timestamp and text
                output = f"[{{segment.start:.2f}}s] {{segment.text}}"
                print(output)
                f.write(output + "\n")
        
        print("--- TRANSCRIPT END ---")
        print(f"\nSUCCESS: Transcript saved to '{{output_file}}'")
        
    except FileNotFoundError as e:
        print(f"Error: File or FFmpeg not found - {{e}}")
    except RuntimeError as e:
        print(f"Transcription error: {{e}}")
    except IOError as e:
        print(f"File I/O error: {{e}}")
    except Exception as e:
        print(f"Unexpected error: {{e}}")

if __name__ == "__main__":
    run_agent()