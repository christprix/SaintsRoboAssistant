from openai import OpenAI
from dotenv import load_dotenv
import os
import sounddevice
from scipy.io.wavfile import write
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

#CREATE OPENAI CLIENT
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

#CREATE AZURE AI SPEAKER

speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SPEECH_REGION')

text = 'Hello sir, how can I help you?'
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Note: the voice setting will not overwrite the voice element in input SSML.
speech_config.speech_synthesis_voice_name = "en-US-DavisNeural"
# use the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

#ASK FOR FOR INPUT
result = speech_synthesizer.speak_text_async(text).get()

#RECORD USER INPUT
#set sample rate
fs = 44100

#input recording time
second = int(input("enter the recording time in seconds: "))
print("begin recording")

#record voice
record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
sounddevice.wait()
print('end recording')
write("Myrecording2.wav", fs, record_voice)

#SPEECH TO TEXT
audio_file= open("./Myrecording2.wav", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcription.text)

#ASK CHATGPT question
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an assistant helping me understand science."},
    {"role": "user", "content": transcription.text}
  ]
)
response = completion.choices[0].message
print(response.content)

#TEXT TO SPEECH

chatgptresponse = speech_synthesizer.speak_text_async(response.content).get()


# result = speech_synthesizer.speak_text_async(text).get()
# # Check result
# if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#     print("Speech synthesized for text [{}]".format(text))
# elif result.reason == speechsdk.ResultReason.Canceled:
#     cancellation_details = result.cancellation_details
#     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#     if cancellation_details.reason == speechsdk.CancellationReason.Error:
#         print("Error details: {}".format(cancellation_details.error_details))
