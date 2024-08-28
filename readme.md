# AI Voice-Based Conversational Pipeline

## Overview

This repository implements a conversational AI pipeline that converts voice input to text, processes it using a Large Language Model (LLM), and then converts the output text back into speech. The pipeline focuses on low latency, effective Voice Activity Detection (VAD), and customizable output features such as pitch, voice type, and speech speed. 

## Pipeline Steps

### 1. Voice-to-Text Conversion

**Objective**: Convert voice input (either from a microphone or audio file) into text.

**Details**:
- **Model**: Use Whisper or its variants for Speech-to-Text (STT) conversion.
  - **Whisper**: [GitHub Repository](https://github.com/openai/whisper)
  - **whisper.cpp**: [GitHub Repository](https://github.com/ggerganov/whisper.cpp)
  - **faster-whisper**: [GitHub Repository](https://github.com/SYSTRAN/faster-whisper)
- **Settings**:
  - **Sampling Rate**: 16 kHz
  - **Audio Channel Count**: 1 (mono)
  - **VAD Threshold**: 0.5 (to detect voice activity and ignore silence)

### 2. Text Input into LLM

**Objective**: Process the converted text query using a pre-trained Large Language Model (LLM).

**Details**:
- **Model**: Choose a suitable pre-trained LLM model from Hugging Face Transformers, such as:
  - **Mistral**: [Hugging Face Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
  - **Llama**, **Mixtral**, **Phi2**, etc.
- **Processing**:
  - The text output from Step 1 is used as a query for the LLM.
  - The response is limited to a maximum of 2 sentences.

### 3. Text-to-Speech Conversion

**Objective**: Convert the LLM-generated text into speech.

**Details**:
- **Model**: Use a Text-to-Speech (TTS) model from the following options:
  - **edge-tts**: [GitHub Repository](https://github.com/rany2/edge-tts)
  - **SpeechT5**: [Hugging Face Model](https://huggingface.co/microsoft/speecht5_tts)
  - **Bark**: [Hugging Face Model](https://huggingface.co/suno/bark)
  - **Parler TTS**: [Hugging Face Model](https://huggingface.co/parler-tts/parler-tts-large-v1)
- **Output Format**: `.mp3` or `.wav`

## Additional Requirements

1. **Latency**: 
   - Minimize latency to below 500 ms using Web Real-Time Communication (WRTC) for efficient audio processing and transmission.

2. **Voice Activity Detection (VAD)**: 
   - Implement VAD to detect voice activity and ignore silence, ensuring efficient audio processing.

3. **Output Restriction**:
   - Restrict the LLM response to a maximum of 2 sentences to maintain clarity and relevance.

4. **Tunable Parameters**:
   - **Pitch**: Adjust the pitch of the synthesized speech.
   - **Voice Type**: Choose between male and female voices (e.g., Joanna or Samantha).
   - **Speed**: Adjust the speed of the synthesized speech.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
2. **Install Dependencies**: Ensure you have Python 3.7+ installed. Install the required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
3. **Running the Pipeline**:
4.   -  **Script**: `Convo_TTS_LLM - Main.py` (Main script for running the pipeline)
     -  **Test Script**: `Convo_TTS_LLM - Test.py` (For testing purposes)
     -  **Jupyter Notebooks**: `AVID.ipynb`, `LLM-TTS.ipynb`, `Transcption_to _LLM(Convo).ipynb` (For exploration and testing)
     -  To run the main script:
        ```bash
        python Convo_TTS_LLM - Main.py --model base --energy_threshold 1000 --record_timeout 1 --phrase_timeout 1.5 --voice "en-US-EmmaNeural" --rate "+1%" --pitch "+1Hz"

  
## Files Description
  - `aud.mp3`: Example audio file in MP3 format.
  - `aud.wav`: Example audio file in WAV format.
  - `AVID.ipynb`: Jupyter notebook for Voice Activity Detection.
  - `Convo_TTS_LLM - Main.py`: Main script for the conversational pipeline.
  - `Convo_TTS_LLM - Test.py`: Test script for verifying the pipeline.
  - `hello.mp3`: Example MP3 file for testing.
  - `requirements.txt`: List of Python dependencies.
  - `Transcption_to _LLM(Convo).ipynb`: Jupyter notebook for transcription and LLM interaction.
  - `Transcption_to _LLM(Convo).py`: Python script for transcription and LLM interaction.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Ensure that any contributions adhere to the project's coding standards and include appropriate tests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements 
  - Whisper model by OpenAI
  - Mistral-7B-Instruct from Hugging Face
  - Edge-TTS for text-to-speech synthesis

For any questions or issues, please contact your-email@example.com.

#
