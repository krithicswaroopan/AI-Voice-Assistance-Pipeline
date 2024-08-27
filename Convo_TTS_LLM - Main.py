import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import webrtcvad
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from huggingface_hub import InferenceClient
import re
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import asyncio
import tempfile


def query_llm(text, client):
    """
    Query the language model and get a response.

    Args:
        text (str): The input text to query.
        client (InferenceClient): The inference client for querying.

    Returns:
        str: The response text limited to 2 sentences.
    """
    response = ""
    try:
        for message in client.chat_completion(
            messages=[{"role": "user", "content": text}],
            max_tokens=500,
            stream=True,
        ):
            response += message.choices[0].delta.content
    except Exception as e:
        print(f"Error querying LLM: {e}")

    # Limit the response to 2 sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    limited_response = ' '.join(sentences[:2])
    return limited_response


def save_to_file(transcriptions, responses, durations):
    """
    Save transcriptions, responses, and durations to a file.

    Args:
        transcriptions (list): List of transcriptions.
        responses (list): List of responses from LLM.
        durations (list): List of processing durations.
    """
    try:
        if not os.path.exists('responses'):
            os.makedirs('responses')

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"responses/{timestamp}_transcriptions_responses.txt"

        with open(filename, 'w') as f:
            max_len = max(len(transcriptions), len(responses), len(durations))
            for i in range(max_len):
                if i < len(transcriptions):
                    f.write(f"Transcription [{i+1}]:\n")
                    f.write(f"{transcriptions[i]}\n")
                if i < len(responses):
                    f.write(f"Response [{i+1}]:\n")
                    f.write(f"{responses[i]}\n")
                if i < len(durations):
                    f.write(f"Processing Time: {durations[i]:.4f}s\n")
                f.write("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"Error saving to file: {e}")


async def text_to_speech(response_text, voice="en-US-EmmaNeural", rate="+1%", pitch="+1Hz"):
    """
    Convert text to speech and play the audio.

    Args:
        response_text (str): The text to convert to speech.
        voice (str): Voice selection.
        rate (str): Speech speed rate.
        pitch (str): Pitch adjustment.
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_filename = temp_audio_file.name
        
        # Initialize TTS and save to the temporary file
        communicate = edge_tts.Communicate(text=response_text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(temp_audio_filename)

        # Play the audio using pydub
        try:
            audio = AudioSegment.from_file(temp_audio_filename, format="mp3")
            play(audio)
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            # Delete the temporary file after playback
            if os.path.exists(temp_audio_filename):
                os.remove(temp_audio_filename)
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")


async def process_audio(args):
    """
    Process audio input, transcribe it, query LLM, and handle responses.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = []
    responses = []
    durations = []

    vad = webrtcvad.Vad(1)

    client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.1",
        token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )

    try:
        while True:
            with source as audio_source:
                print("Listening for audio...")
                try:
                    audio_data = recorder.listen(audio_source, timeout=record_timeout)
                except sr.WaitTimeoutError:
                    print("No speech detected. Retrying...")
                    continue

                audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

                # Apply VAD
                frame_duration_ms = 30
                sample_rate = 16000
                frame_size = int(sample_rate * frame_duration_ms / 1000)
                num_frames = len(audio_np) // frame_size

                contains_speech = False
                for i in range(num_frames):
                    frame = audio_np[i * frame_size:(i + 1) * frame_size]
                    frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                    if vad.is_speech(frame_bytes, sample_rate=sample_rate):
                        contains_speech = True
                        break

                if not contains_speech:
                    sleep(0.05)
                    continue

                start_time = datetime.now()

                # Transcribe the audio
                try:
                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    continue

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                transcription.append(f"[{duration:.4f}s] {text}")

                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n".join(transcription))
                print('', end='', flush=True)

                try:
                    response = query_llm(text, client)
                    responses.append(response)
                    durations.append(duration)
                except Exception as e:
                    print(f"Error during LLM query: {e}")

                print("\nLLM Response:")
                print(response)

                # Convert the response text to speech
                try:
                    await text_to_speech(response, voice=args.voice, rate=args.rate, pitch=args.pitch)
                except Exception as e:
                    print(f"Error during text-to-speech: {e}")

                # Calculate quit period
                quit_period = duration # Adding 5 seconds as buffer
                sleep(quit_period)

    except KeyboardInterrupt:
        print("Exiting gracefully...")
    finally:
        save_to_file(transcription, responses, durations)
        print("\n\nFinal Transcription:")
        print("\n".join(transcription))


def main():
    """
    Main function to parse arguments and start the audio processing.
    """
    parser = argparse.ArgumentParser(description="Real-time audio processing and LLM interaction.")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size.")
    parser.add_argument("--non_english", action='store_true',
                        help="Use non-English model variant.")
    parser.add_argument("--energy_threshold", default=1000, type=int,
                        help="Energy threshold for speech detection.")
    parser.add_argument("--record_timeout", default=5, type=float,
                        help="Timeout for listening in seconds.")
    parser.add_argument("--phrase_timeout", default=1.5, type=float,
                        help="Timeout for detecting a complete phrase.")
    parser.add_argument("--voice", default="en-US-EmmaNeural",
                        help="Voice selection for text-to-speech.")
    parser.add_argument("--rate", default="+1%", help="Speech speed rate.")
    parser.add_argument("--pitch", default="+1Hz", help="Pitch adjustment.")
    
    args = parser.parse_args()

    asyncio.run(process_audio(args))


if __name__ == "__main__":
    main()
