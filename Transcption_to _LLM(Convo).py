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

def query_llm(text, client):
    response = ""
    for message in client.chat_completion(
        messages=[{"role": "user", "content": text}],
        max_tokens=500,
        stream=True,
    ):
        response += message.choices[0].delta.content
    
    # Limit the response to 2 sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    limited_response = ' '.join(sentences[:2])
    return limited_response

def save_to_file(transcriptions, responses, durations):
    # Create the responses directory if it does not exist
    if not os.path.exists('responses'):
        os.makedirs('responses')

    # Generate filename with date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"responses/{timestamp}_transcriptions_responses.txt"

    # Write to file
    with open(filename, 'w') as f:
        for i in range(len(transcriptions)):
            f.write(f"Transcription [{i+1}]:\n")
            f.write(f"{transcriptions[i]}\n")
            f.write(f"Response [{i+1}]:\n")
            f.write(f"{responses[i]}\n")
            f.write(f"Processing Time: {durations[i]:.4f}s\n")
            f.write("\n" + "-"*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true')
    parser.add_argument("--energy_threshold", default=1000, type=int)
    parser.add_argument("--record_timeout", default=1, type=float)
    parser.add_argument("--phrase_timeout", default=1.5, type=float)
    args = parser.parse_args()

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

    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(1)  # Mode 1 is a good balance for sensitivity

    # Initialize the Hugging Face Inference Client
    client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.1",
        token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True

                phrase_time = now

                # Process each frame of audio data
                while not data_queue.empty():
                    audio_data = data_queue.get()

                    # Convert raw PCM data to a numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Convert float32 to int16 for VAD processing
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_pcm = audio_int16.tobytes()

                    # Apply VAD
                    frame_duration_ms = 30
                    sample_rate = 16000
                    frame_size = int(sample_rate * frame_duration_ms / 1000)
                    num_frames = len(audio_pcm) // (frame_size * 2)

                    contains_speech = False
                    for i in range(num_frames):
                        frame = audio_pcm[i * frame_size * 2:(i + 1) * frame_size * 2]
                        if vad.is_speech(frame, sample_rate=sample_rate):
                            contains_speech = True
                            break

                    if not contains_speech:
                        continue

                    # Track the start time
                    start_time = datetime.now()

                    # Transcribe the audio frame
                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # Track the end time
                    end_time = datetime.now()

                    # Calculate the time taken for processing
                    duration = (end_time - start_time).total_seconds()

                    # Append the new transcription to the list
                    transcription.append(f"[{duration:.4f}s] {text}")

                    # Clear the console and print the updated transcription
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("\n".join(transcription))
                    print('', end='', flush=True)

                    # Query LLM with the transcribed text
                    response = query_llm(text, client)
                    responses.append(response)
                    durations.append(duration)
                    
                    print("\nLLM Response:")
                    print(response)

            else:
                sleep(0.05)  # Reduced sleep time for improved responsiveness

        except KeyboardInterrupt:
            break

    # Save to file after exiting the loop
    save_to_file(transcription, responses, durations)

    print("\n\nFinal Transcription:")
    print("\n".join(transcription))

if __name__ == "__main__":
    main()
