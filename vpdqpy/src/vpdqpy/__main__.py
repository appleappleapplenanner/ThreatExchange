import logging
from typing import Optional, Annotated, List
from pathlib import Path

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


class PDQFrameBufferHasher:
    """
    //  Variables for computing pdq hash
    std::vector<float> _fullLumaImageBuffer1;
    std::vector<float> _fullLumaImageBuffer2;
    static const int SCALED_DIMENSION = 64;
    float _buffer64x64[64][64];
    float _buffer16x64[16][64];
    float _buffer16x16[16][16];
    """
    SCALED_DIMENSION: int = 64
    FEATURE_DIMENSION: int = 256
    MIN_HASHABLE_DIM: int = 5
    
    _fullLumaImageBuffer1: List[float] = [] # Size is _numRGBTriples
    _fullLumaImageBuffer2: List[float] = [] # Size is _numRGBTriples
    _buffer64x64: List[List] = [[]]
    _buffer16x64: List[List] = [[]]
    _buffer16x16: List[List] = [[]]

    def __init__(self, frameHeight: int, frameWidth: int):
        self._frameHeight = frameHeight
        self._frameWidth = frameWidth
        self._numRGBTriples = frameHeight * frameWidth

    """
    // Get PDQ Hash in Hash256 format
    bool PDQFrameBufferHasher::hashFrame(
        unsigned char* buffer,
        pdq::hashing::Hash256& hash, // The result pdq hash
        int& quality // Hashing Quality
    ) {
    if (_frameHeight < MIN_HASHABLE_DIM || _frameWidth < MIN_HASHABLE_DIM) {
        hash.clear();
        quality = 0;
        return false;
    }
    facebook::pdq::hashing::fillFloatLumaFromRGB(
        &buffer[0], // pRbase
        &buffer[1], // pGbase
        &buffer[2], // pBbase
        _frameHeight,
        _frameWidth,
        3 * _frameWidth, // rowStride
        3, // colStride
        _fullLumaImageBuffer1.data());

    facebook::pdq::hashing::pdqHash256FromFloatLuma(
        _fullLumaImageBuffer1.data(), // numRows x numCols, row-major
        _fullLumaImageBuffer2.data(), // numRows x numCols, row-major
        _frameHeight,
        _frameWidth,
        _buffer64x64,
        _buffer16x64,
        _buffer16x16,
        hash,
        quality);

    return true;
    }

    };
    """

def get_vid_info(video: Path | str):
    probe = ffmpeg.probe(video)
    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if not video_info:
        raise ValueError("No video stream found in the input file.")
    return video_info

def main():
    pdq = PDQHasher()
    
    testdir = Path(__file__).parent.parent.parent / "tests/vids"
    testpic = testdir / "grass.jpeg"
    testvid1 = testdir / "Big_Buck_Bunny_720_10s_1MB.mp4"
    testpic_pdq = pdq.fromFile(filepath=str(testpic))
    print(testpic_pdq.getHash())


    # Example usage:
    interval = 1  # Extract a frame every 10 seconds

    out, _= (
            ffmpeg
            .input(testvid1)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1)
            .run(capture_stdout=True)
        )
    

    video_info = get_vid_info(testvid1)
    print(video_info)
    width = int(video_info['width'])
    height = int(video_info['height'])        
    framerate = float(eval('+'.join(video_info['avg_frame_rate'].split())))
    duration = float(video_info['duration'])
    print(f"{framerate} FPS, {duration} seconds")
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    print(len(video))
    for i, frame in enumerate(video):
        image = Image.fromarray(frame)
        print(pdq.fromBufferedImage(image).hash)
        timestamp = (i / duration) * framerate
        print(timestamp)
        #image.save(f"frame{i}.jpg", format="JPEG")


    #image.save("frame.jpg", format="JPEG")
    typer.Exit()

try:
    typer.run(main)
except KeyboardInterrupt:
    typer.Exit()