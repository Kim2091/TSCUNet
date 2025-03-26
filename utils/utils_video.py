import gc
import av
import threading
import queue

from fractions import Fraction

class VideoDecoder(threading.Thread):
    def __init__(self, input_path, options={}):
        super().__init__()

        # Open the input file and get the video stream
        self.input_container = av.open(input_path, options=options)
        self.input_stream = self.input_container.streams.video[0]
        self.input_stream.thread_type = 'AUTO'

        # Approximate number of frames with a couple seconds of headroom
        self.frame_count = int((self.input_container.duration/1000000 + 10)*23.976)

        # Create a queue to hold the frames that will be decoded
        self.frame_queue = queue.Queue(3)

        # Set a flag to indicate whether the thread should continue running
        self.running = True

    def run(self):
        # Keep getting frames until the running flag is set to False
        while self.running:
            try:
                for frame in self.input_container.decode(video=0):
                    self.frame_queue.put(frame.to_ndarray(format='rgb24'), block=True)
            except av.error.EOFError:
                self.running = False
            
        self.input_container.close()

    def get_frame(self):
        # Get a frame from the queue
        try:
            return self.frame_queue.get(block=True, timeout=1)
        except queue.Empty:
            return None

    def stop(self):
        # Set the running flag to False to stop the thread
        self.running = False
    
    def __len__(self):
        return self.frame_count

class VideoEncoder(threading.Thread):
    def __init__(self, output_path, width, height, fps=Fraction(24000, 1001), codec='libx264', pix_fmt='yuv420p', options={}, input_depth=8):
        super().__init__()

        # Create a video container and stream with the specified codec and parameters
        self.output_container = av.open(output_path, mode='w')
        self.stream = self.output_container.add_stream(codec, rate=fps, options=options)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt

        self.input_depth = input_depth

        # Create a queue to hold the frames that will be encoded
        self.frame_queue = queue.Queue(3)

        # Set a flag to indicate whether the thread should continue running
        self.running = True

    def run(self):
        # Keep encoding frames until the running flag is set to False
        while self.running:
            # Try to get a frame from the queue
            try:
                frame = self.frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Encode the frame
            frame = av.VideoFrame.from_ndarray(frame, format="rgb48le" if self.input_depth == 16 else "rgb24")
            for packet in self.stream.encode(frame):
                self.output_container.mux(packet)
            
            del frame
            gc.collect()

        # Flush the encoder and close the output file when the thread is finished
        for packet in self.stream.encode():
            self.output_container.mux(packet)
        self.output_container.close()

    def add_frame(self, frame):
        # Add a frame to the queue
        self.frame_queue.put(frame, block=True)

    def stop(self):
        # Set the running flag to False to stop the thread
        self.running = False