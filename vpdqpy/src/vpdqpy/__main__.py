import logging
from typing import Optional, Annotated, List
from pathlib import Path
import subprocess
import json

from pdqhashing.hasher.pdq_hasher import PDQHasher
from pdqhashing.types.hash256 import Hash256
from pdqhashing.types.containers import HashAndQuality

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
class VpdqFeature:
    pdqHash: Hash256 # 64 char hex string
    quality: float # 0 to 100
    frameNumber: int
    timeStamp: float  # This is frameNumber / framesPerSec


def get_vid_info(file: bytes) -> dict:
    # ffprobe command to get info. ffmpeg-python requires a file name, this does not.
    ffprobe_process = subprocess.Popen(
        ["ffprobe", "-show_streams", "-print_format", "json", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Retrieve the output and error streams from ffprobe
    stdout, stderr = ffprobe_process.communicate(input=file)

    # Decode the output stream as json
    output = json.loads(stdout.decode("utf-8"))
    error = stderr.decode("utf-8")

    video_info = next(
        (stream for stream in output["streams"] if stream["codec_type"] == "video"),
        None,
    )
    if not video_info:
        raise ValueError("No video stream found in the input file.")
    return video_info


# Get the bytes of a video
def get_video_bytes(video_file: Path | str | bytes) -> bytes:
    video: bytes = None
    if isinstance(video_file, (Path, str)):
        try:
            with open(str(video_file), "rb") as file:
                video = file.read()
        except OSError:
            raise ValueError("Failed to get video file bytes. Invalid object type.")
    elif isinstance(video_file, bytes):
        video = video_file
    else:
        raise ValueError("Failed to get video file bytes. Invalid object type.")

    return video

# Filter out the VPDQ feature with exact same hash
def dedupe_features(features: list[VpdqFeature]) -> list[VpdqFeature]:
    unique_features = set()
    ret = []
    for feature in features:
        if str(feature.pdqHash) not in unique_features:
            ret.append(feature)
            unique_features.add(str(feature.pdqHash))
    return ret

# quality tolerance from [0,100]
def filter_features(
    vpdq_features: list[VpdqFeature], quality_tolerance: float
) -> list[VpdqFeature]:
    return [
        feature
        for feature in vpdq_features
        if feature.quality >= quality_tolerance
    ]

# Get number of matching features for query and target
def feature_match_count(
    query_features: list[VpdqFeature],
    target_features: list[VpdqFeature],
    distance_tolerance: float,
) -> int:
    return sum(
        any(
            query_feature.pdqHash.hammingDistance( target_feature.pdqHash ) <= distance_tolerance
            for target_feature in target_features
        )
        for query_feature in query_features
    )


def match_hash(
    query_features: list[VpdqFeature],
    target_features: list[VpdqFeature],
    quality_tolerance: float = 50,
    distance_tolerance: float = 31,
):
    query_filtered = filter_features(dedupe_features(query_features), quality_tolerance)
    target_filtered = filter_features(dedupe_features(target_features), quality_tolerance)

    # Avoid divide by zero
    if len(query_filtered) <= 0 or len(target_filtered) <= 0:
        return 0

    result = feature_match_count(query_filtered, target_filtered, distance_tolerance)
    return result * 100 / len(query_filtered)


# Perceptually hash video from a file path or the bytes
def computeHash(video_file: Path | str | bytes) -> list[VpdqFeature]:
    video = get_video_bytes(video_file)
    if video is None:
        return

    video_info = get_vid_info(video)
    # print(video_info)
    width = int(video_info["width"])
    height = int(video_info["height"])
    framerate = float(eval("+".join(video_info["avg_frame_rate"].split())))
    duration = float(video_info["duration"])
    print(f"{framerate} FPS, {duration} seconds")

    interval = 1 / 1  # How often to get a frame in seconds / frame

    out, _ = (
        ffmpeg.input("pipe:")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=interval)
        .run(input=video, capture_stdout=True)
    )

    video_frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    pdq = PDQHasher()
    features: list[VpdqFeature] = []
    for frameNum, frame in enumerate(tqdm(video_frames)):
        image = Image.fromarray(frame)
        pdqHashAndQuality = pdq.fromBufferedImage(image)
        timestamp = 10 * (frameNum) / (duration * interval)  # in seconds
        pdqFrame = VpdqFeature(pdqHashAndQuality.getHash(), pdqHashAndQuality.getQuality(), frameNum, timestamp) 
        features.append(pdqFrame)

    return features

def main():
    testdir = Path(__file__).parent.parent.parent / "tests/videos"
    testvids = [
        "Big_Buck_Bunny_720_10s_1MB.mp4",
        "Big_Buck_Bunny_1080_10s_1MB.mp4",
        "Jellyfish_360_10s_1MB.mp4",
        "Jellyfish_1080_10s_1MB.mp4",
        "Sintel_360_10s_1MB.mp4",
        "Sintel_1080_10s_1MB.mp4",
    ]
    testvids = [testdir / testvid for testvid in testvids]
    testvidbytes = []
    # Read the file into bytes
    for vid in testvids:
        with open(vid, "rb") as file:
            video_bytes = file.read()
        testvidbytes.append(video_bytes)

    testvidphash = [computeHash(testvidbyte) for testvidbyte in testvidbytes]

    for i in range(0, len(testvidphash)):
        if i + 1 < len(testvidphash):
            print(
                f"Match: {i}, {i+1}", match_hash(testvidphash[i], testvidphash[i + 1])
            )

    typer.Exit()


try:
    typer.run(main)
except KeyboardInterrupt:
    typer.Exit()
