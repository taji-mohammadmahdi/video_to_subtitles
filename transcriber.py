from faster_whisper import WhisperModel
from pathlib import Path

# Initialize the speech recognition model
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# Specify the input video file path
video_path = Path("video.mp4")

# Transcribe speech from the video
segments, info = model.transcribe(
    str(video_path),
    language="en",
    beam_size=5
)

# Prepare lists for text and subtitle output
txt_output = []
srt_output = []

for index, segment in enumerate(segments, start=1):

    text = segment.text.strip()

    txt_output.append(text)

    def timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return (
            f"{hours:02}:"
            f"{minutes:02}:"
            f"{seconds_part:02},"
            f"{milliseconds:03}"
        )

    srt_output.append(
        f"{index}\n"
        f"{timestamp(segment.start)} --> "
        f"{timestamp(segment.end)}\n"
        f"{text}\n"
    )

# Save the transcription as a TXT file
Path("transcript.txt").write_text(
    "\n".join(txt_output),
    encoding="utf-8"
)

# Save the subtitles as an SRT file
Path("subtitles.srt").write_text(
    "\n".join(srt_output),
    encoding="utf-8"
)

print("Transcription completed!")
print(f"Detected language: {info.language}")
print("Files saved: transcript.txt and subtitles.srt")