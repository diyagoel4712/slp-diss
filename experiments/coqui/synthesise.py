from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="The rain in Spain stays mainly in the plain.",
    speaker_wav="../../vctk/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_001_mic1.flac",
    language="en",
    file_path="output.wav"
)

