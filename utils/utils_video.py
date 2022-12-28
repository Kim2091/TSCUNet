import av
import threading
import queue

from fractions import Fraction

def get_codec_options(codec):
    pix_fmt = 'yuv444p10le'
    options = { 'c': codec }

    if codec == 'dnxhd':
        options['profile'] = 'dnxhr_444'
    elif codec == 'libx264':
        options['profile'] = 'high444'
        options['crf'] = '13'
        options['preset'] = 'slow'
    elif codec == 'libx265':
        options['profile'] = 'main444-12'
        options['crf'] = '13'
        options['preset'] = 'slow'
        pix_fmt = 'yuv444p12le'

    return pix_fmt, options

class VideoEncoder(threading.Thread):
    def __init__(self, output_path, width, height, fps=Fraction(24000, 1001), codec='libx264', pix_fmt='yuv444p10le', options={}, input_depth=8):
        super().__init__()

        # Create a video container and stream with the specified codec and parameters
        self.output_container = av.open(output_path, mode='w')
        self.stream = self.output_container.add_stream(codec, rate=fps, options=options)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt

        self.input_depth = input_depth

        # Create a queue to hold the frames that will be encoded
        self.frame_queue = queue.Queue()

        # Set a flag to indicate whether the thread should continue running
        self.running = True

    def run(self):
        # Keep encoding frames until the running flag is set to False
        while self.running:
            # Try to get a frame from the queue
            frame = self.frame_queue.get(block=True)

            # Encode the frame
            for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb48le" if self.input_depth == 16 else "rgb24")):
                self.output_container.mux(packet)

        # Flush the encoder and close the output file when the thread is finished
        for packet in self.stream.encode():
            self.output_container.mux(packet)
        self.output_container.close()

    def add_frame(self, frame):
        # Add a frame to the queue
        self.frame_queue.put(frame)

    def stop(self):
        # Set the running flag to False to stop the thread
        self.running = False