import logging
from typing import Optional, Annotated, List
from pathlib import Path
import subprocess
import json

from pdqhashing.hasher.pdq_hasher import PDQHasher

import typer
from rich import print as rprint

from .__about__ import __version__

from dataclasses import dataclass
from typing import List
import ffmpeg
import numpy as np
from PIL import Image
import io
from tqdm import tqdm

@dataclass(slots=True)
class vpdqFeature:
    hash256: PDQHasher
    frameNumber: int
    quality: int
    timeStamp: float # This is frameNumber / framesPerSec

@dataclass(slots=True)
class PDQFrame:
    pdqHash: str
    frameNumber: int
    quality: int
    timeStamp: float # This is frameNumber / framesPerSec

def get_vid_info(file: bytes) -> dict:
    # ffprobe command to get info. ffmpeg-python requires a file name, this does not.
    ffprobe_process = subprocess.Popen(
        ['ffprobe', '-show_streams', '-print_format', 'json', '-'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Retrieve the output and error streams from ffprobe
    stdout, stderr = ffprobe_process.communicate(input=file)

    # Decode the output stream as json
    output = json.loads(stdout.decode('utf-8'))
    error = stderr.decode('utf-8')

    video_info = next((stream for stream in output['streams'] if stream['codec_type'] == 'video'), None)
    if not video_info:
        raise ValueError("No video stream found in the input file.")
    return video_info

# Get the bytes of a video
def get_video_bytes(video_file: Path | str | bytes) -> bytes:
    video: bytes = None
    if isinstance(video_file, (Path, str)):
        try:
            with open(str(video_file), 'rb') as file:
                video = file.read()
        except OSError:
            raise ValueError("Failed to get video file bytes. Invalid object type.")
    elif isinstance(video_file, bytes):
        video = video_file
    else:
        raise ValueError("Failed to get video file bytes. Invalid object type.")

    return video

# Perceptually hash video from a file path or the bytes
def phash_video(video_file: Path | str | bytes) -> list[PDQFrame]:
    video = get_video_bytes(video_file)
    if video is None:
        return
    
    video_info = get_vid_info(video)
    print(video_info)
    width = int(video_info['width'])
    height = int(video_info['height'])        
    framerate = float(eval('+'.join(video_info['avg_frame_rate'].split())))
    duration = float(video_info['duration'])
    print(f"{framerate} FPS, {duration} seconds")

    interval = 1/1 # How often to get a frame in seconds / frame

    out, _= (
            ffmpeg
            .input('pipe:')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=interval)
            .run(input=video, capture_stdout=True)
        )
    

    video_frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    pdq = PDQHasher()
    pdqHashes: list[PDQFrame] = []
    for frameNum, frame in enumerate(tqdm(video_frames)):
        image = Image.fromarray(frame)
        pdqHashAndQuality = pdq.fromBufferedImage(image)
        timestamp = 10 * (frameNum) / (duration * interval) # in seconds
        pdqFrame = PDQFrame(pdqHashAndQuality.hash, frameNum,
                            pdqHashAndQuality.quality, timestamp)
        pdqHashes.append(pdqFrame)

    return pdqHashes

def main():
    
    testdir = Path(__file__).parent.parent.parent / "tests/vids"
    testvid1 = testdir / "Big_Buck_Bunny_720_10s_1MB.mp4"
    # Read the file into bytes
    with open(testvid1, 'rb') as file:
        video_bytes = file.read()

    print(phash_video(video_bytes))

    typer.Exit()

try:
    typer.run(main)
except KeyboardInterrupt:
    typer.Exit()