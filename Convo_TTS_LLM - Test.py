import argparse  # Import argparse for command-line argument parsing
import os  # Import os for interacting with the operating system
import numpy as np  # Import numpy for numerical operations
import speech_recognition as sr  # Import speech_recognition for speech-to-text
import whisper  # Import whisper for transcription
import torch  # Import torch for tensor computations
import webrtcvad  # Import webrtcvad for voice activity detection
from datetime import datetime, timedelta  # Import datetime and timedelta for time operations
from queue import Queue  # Import Queue for thread-safe data storage
from time import sleep  # Import sleep for pausing execution
from huggingface_hub import InferenceClient  # Import InferenceClient for querying LLM
import re  # Import re for regular expression operations
import edge_tts  # Import edge_tts for text-to-speech conversion
from pydub import AudioSegment  # Import AudioSegment for audio file handling
from pydub.playback import play  # Import play for audio playback
import asyncio  # Import asyncio for asynchronous operations

def query_llm(text, client):
    """
    Queries the LLM model and returns a response limited to 2 sentences.

    Args:
        text (str): The text input for the LLM.
        client (InferenceClient): The InferenceClient instance for interacting with the LLM.

    Returns:
        str: The LLM response limited to 2 sentences.
    """
    response = ""  # Initialize an empty string for the LLM response
    try:
        # Query the LLM model with the input text and stream the response
        for message in client.chat_completion(
            messages=[{"role": "user", "content": text}],  # Set the input message for the LLM
            max_tokens=500,  # Set the maximum number of tokens for the response
            stream=True,  # Enable streaming mode
        ):
            response += message.choices[0].delta.content  # Append each chunk of the response to the full response
    except Exception as e:
        print(f"Error querying LLM: {e}")  # Print error message if querying fails
        return ""

    # Split the response into sentences and limit to 2 sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)  # Split response by sentence delimiters
    limited_response = ' '.join(sentences[:2])  # Join the first 2 sentences
    return limited_response  # Return the limited response

def save_to_file(transcriptions, responses, durations):
    """
    Saves transcriptions, responses, and durations to a text file.

    Args:
        transcriptions (list): List of transcriptions.
        responses (list): List of LLM responses.
        durations (list): List of processing durations.
    """
    # Create a directory for saving responses if it doesn't exist
    if not os.path.exists('responses'):
        os.makedirs('responses')

    # Generate a filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"responses/{timestamp}_transcriptions_responses.txt"

    try:
        # Open the file for writing
        with open(filename, 'w') as f:
            # Write transcriptions, responses, and durations to the file
            for i in range(len(transcriptions)):
                f.write(f"Transcription [{i+1}]:\n")  # Write the transcription index
                f.write(f"{transcriptions[i]}\n")  # Write the transcription text
                f.write(f"Response [{i+1}]:\n")  # Write the response index
                f.write(f"{responses[i]}\n")  # Write the LLM response
                f.write(f"Processing Time: {durations[i]:.4f}s\n")  # Write the processing duration
                f.write("\n" + "-"*50 + "\n")  # Add a separator line
    except IOError as e:
        print(f"Error saving to file: {e}")  # Print error message if file saving fails

async def text_to_speech(response_text, voice="en-US-EmmaNeural", rate="+1%", pitch="+1Hz"):
    """
    Converts text to speech and plays it.

    Args:
        response_text (str): The text to be converted to speech.
        voice (str): The voice to be used for TTS.
        rate (str): The speech rate adjustment.
        pitch (str): The pitch adjustment.
    """
    # Create a directory for saving audio responses if it doesn't exist
    if not os.path.exists('audio_responses'):
        os.makedirs('audio_responses')

    # Generate an audio filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audio_filename = f"audio_responses/{timestamp}_response.mp3"

    # Initialize TTS
    communicate = edge_tts.Communicate(text=response_text, voice=voice, rate=rate, pitch=pitch)

    try:
        # Save the output to an mp3 file
        await communicate.save(audio_filename)

        # Check if the file was created
        if not os.path.isfile(audio_filename):
            print(f"File not found: {audio_filename}")  # Print error message if file not found
            return

        # Print the file path for debugging
        print(f"Playing audio from: {audio_filename}")

        # Play the audio using pydub
        try:
            audio = AudioSegment.from_mp3(audio_filename)  # Load the mp3 file
            play(audio)  # Play the audio file
        except Exception as e:
            print(f"Error playing sound: {e}")  # Print error message if playback fails

    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")  # Print error message if TTS fails

async def process_audio(args):
    """
    Processes audio input from the microphone, performs transcription, queries LLM, and handles responses.

    Args:
        args (Namespace): Command-line arguments.
    """
    phrase_time = None  # Initialize time of the last phrase
    data_queue = Queue()  # Initialize a queue for audio data
    recorder = sr.Recognizer()  # Create a speech recognizer instance
    recorder.energy_threshold = args.energy_threshold  # Set energy threshold for speech recognition
    recorder.dynamic_energy_threshold = False  # Disable dynamic energy threshold adjustment

    source = sr.Microphone(sample_rate=16000)  # Set up the microphone source with a sample rate of 16 kHz

    model = args.model  # Get the model type from command-line arguments
    if args.model != "large" and not args.non_english:
        model += ".en"  # Append ".en" for English models if needed
    
    try:
        audio_model = whisper.load_model(model)  # Load the Whisper model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")  # Print error message if model loading fails
        return

    record_timeout = args.record_timeout  # Get record timeout from command-line arguments
    phrase_timeout = args.phrase_timeout  # Get phrase timeout from command-line arguments

    transcription = []  # Initialize a list to store transcriptions
    responses = []  # Initialize a list to store LLM responses
    durations = []  # Initialize a list to store processing durations

    vad = webrtcvad.Vad(1)  # Initialize Voice Activity Detection with aggressive mode

    try:
        client = InferenceClient(
            "mistralai/Mistral-7B-Instruct-v0.1",  # Set the model to use
            token="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",  # Set the API token
        )
    except Exception as e:
        print(f"Error initializing InferenceClient: {e}")  # Print error message if client initialization fails
        return

    def record_callback(_, audio: sr.AudioData):
        """
        Callback function for handling recorded audio.

        Args:
            _: Unused argument.
            audio (sr.AudioData): Recorded audio data.
        """
        data = audio.get_raw_data()  # Get raw audio data from the AudioData object
        data_queue.put(data)  # Add the audio data to the queue

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)  # Start background listening

    print("Model loaded.\n")  # Notify that the model has been loaded

    while True:
        try:
            now = datetime.utcnow()  # Get the current time in UTC

            if not data_queue.empty():  # Check if there is data in the queue
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True  # Mark the phrase as complete if timeout has passed

                phrase_time = now  # Update the time of the last phrase

                while not data_queue.empty():  # Process all items in the queue
                    audio_data = data_queue.get()  # Get audio data from the queue

                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Convert audio data to numpy array
                    audio_int16 = (audio_np * 32767).astype(np.int16)  # Convert float32 back to int16
                    audio_pcm = audio_int16.tobytes()  # Convert numpy array to bytes

                    frame_duration_ms = 30  # Set frame duration in milliseconds
                    sample_rate = 16000  # Set sample rate
                    frame_size = int(sample_rate * frame_duration_ms / 1000)  # Calculate frame size
                    num_frames = len(audio_pcm) // (frame_size * 2)  # Calculate number of frames

                    contains_speech = False
                    for i in range(num_frames):
                        frame = audio_pcm[i * frame_size * 2:(i + 1) * frame_size * 2]  # Extract frame
                        if vad.is_speech(frame, sample_rate=sample_rate):  # Check if the frame contains speech
                            contains_speech = True
                            break

                    if not contains_speech:  # Skip processing if no speech detected
                        continue

                    start_time = datetime.now()  # Record the start time for processing

                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())  # Perform transcription
                    text = result['text'].strip()  # Get the transcribed text

                    end_time = datetime.now()  # Record the end time for processing

                    duration = (end_time - start_time).total_seconds()  # Calculate processing duration

                    transcription.append(f"[{duration:.4f}s] {text}")  # Append transcription to the list

                    if phrase_complete:
                        limited_response = query_llm(text, client)  # Query LLM with the transcribed text
                        responses.append(limited_response)  # Append LLM response to the list
                        durations.append(duration)  # Append processing duration to the list

                        print(f"\nResponse: {limited_response}\n")  # Print the LLM response

                        asyncio.run(text_to_speech(limited_response))  # Convert response text to speech and play it

                    print("\n".join(transcription))  # Print the updated transcription
                    print('', end='', flush=True)  # Flush the output buffer

            else:
                sleep(0.05)  # Sleep briefly to avoid high CPU usage

        except KeyboardInterrupt:  # Handle keyboard interrupt (Ctrl+C)
            break

    print("\n\nFinal Transcription:")  # Print final transcription header
    print("\n".join(transcription))  # Print final transcription

    # Save the results to a file
    save_to_file(transcription, responses, durations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time speech-to-text transcription with LLM querying and TTS.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (default: base)")
    parser.add_argument("--energy_threshold", type=int, default=1000, help="Energy threshold for speech recognition (default: 1000)")
    parser.add_argument("--record_timeout", type=float, default=1, help="Timeout for recording phrases (default: 1)")
    parser.add_argument("--phrase_timeout", type=float, default=1.5, help="Timeout for detecting phrase completion (default: 1.5)")
    parser.add_argument("--non_english", action='store_true', help="Use non-English models if specified")

    args = parser.parse_args()  # Parse command-line arguments
    asyncio.run(process_audio(args))  # Run the audio processing function
