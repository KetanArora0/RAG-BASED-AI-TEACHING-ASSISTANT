import whisper
import json

model = whisper.load_model("small")

result = model.transcribe(audio = "audios/13.00__13.00_sample.mp4.mp3",
                          language = "hi",
                          task = "translate",
                          word_timestamps = False)


chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)