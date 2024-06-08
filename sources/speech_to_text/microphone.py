import logging
import os
import wave
from typing import Optional, Callable

import pyaudio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MicrophoneRecorder:
    """
    Class to record audio from a microphone using PyAudio.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 1, device_index: int = None):
        """
        Initialize the MicrophoneRecorder.

        :param sample_rate: Sample rate of the recording.
        :param channels: Number of audio channels.
        :param device_index: Index of the audio device to use.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.channels = self.check_channels(channels)
        self.stream = None

        if device_index is not None:
            if not self.is_valid_device_index(device_index):
                logging.warning(f"Device index {device_index} is not available. Using default device.")
                self.device_index = None

    def check_microphone(self) -> bool:
        """
        Check if a microphone is available.

        :return: True if a microphone is available, False otherwise.
        """
        count = self.audio.get_device_count()
        for i in range(count):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                return True
        return False

    def check_channels(self, requested_channels: int) -> int:
        """
        Check the available number of channels and return the appropriate number.

        :param requested_channels: Number of audio channels requested.
        :return: Number of channels that will be used.
        """
        if self.device_index is not None:
            device_info = self.audio.get_device_info_by_index(self.device_index)
        else:
            device_info = self.audio.get_default_input_device_info()

        max_channels = device_info.get('maxInputChannels')

        if max_channels >= requested_channels:
            logging.info(f"Using {requested_channels} channel(s) as requested.")
            return requested_channels
        else:
            logging.warning(
                f"Requested {requested_channels} channel(s), but only {max_channels} channel(s) are available. "
                f"Using {max_channels} channel(s).")
            return max_channels

    def display_devices_info(self):
        """
        Display information about the available microphones.
        """
        count = self.audio.get_device_count()
        for i in range(count):
            device_info = self.audio.get_device_info_by_index(i)
            logging.info(f"Device {i}: {device_info['name']} - Max Input Channels: {device_info['maxInputChannels']}")

    def is_valid_device_index(self, index: int) -> bool:
        """
        Check if the given device index is valid.

        :param index: Device index to check.
        :return: True if the device index is valid, False otherwise.
        """
        count = self.audio.get_device_count()
        return 0 <= index < count

    def start_stream(self, callback: Callable = None, save_path: Optional[str] = None) -> None:
        """
        Start the audio stream, either recording to a file or streaming with a callback.

        :param callback: Function to call with each chunk of audio data if streaming.
        """
        # Check if a microphone is available
        if not self.check_microphone():
            logging.error("No microphone is available. Please check your microphone connection and try again.")
            return

        # Get the device information
        if self.device_index is not None:
            device_info = self.audio.get_device_info_by_index(self.device_index)
            logging.info(f"Selected device: {device_info['name']}")
        else:
            device_info = self.audio.get_default_input_device_info()
            logging.info(f"Using default device: {device_info['name']}")

        # Start the audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=1024
        )

        logging.info("Audio stream started. Press Ctrl+C to stop.")

        frames = []

        try:
            while True:
                data = self.stream.read(1024)
                frames.append(data)
                if callback:
                    callback(data)
        except KeyboardInterrupt:
            logging.info("Streaming stopped.")
        finally:
            self.stop_stream()
            if not callback:
                logging.info("Recording finished.")
                self.save_audio(frames, save_path if save_path else "data/audios")

    def save_audio(self, frames: list, path: str) -> None:
        """
        Save the recorded audio frames to a file.

        :param frames: List of audio frames.
        :param path: Directory path or file path to save the audio file.
        """
        if os.path.isdir(path):
            if not os.path.exists(path):
                os.makedirs(path)
                logging.info(f"Created directory: {path}")

            output_file = f"{path}/output.wav"
        else:
            output_file = path

        # Save the audio frames to a WAV file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        logging.info(f"Audio recorded and saved to {output_file}")

    def stop_stream(self) -> None:
        """
        Stop the audio stream.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logging.info("Audio stream stopped.")

    def __del__(self):
        """
        Clean up the PyAudio instance.
        """
        self.audio.terminate()


def example_callback(data):
    """
    Example callback function to process streamed audio data.

    :param data: Chunk of audio data.
    """
    logging.info("Received audio data chunk.")


if __name__ == "__main__":
    choice_device = 2  # Specify the device index if needed
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Display microphone information
    # recorder.display_devices_info() # Uncomment to display microphone information for debugging

    # To record audio to a file, start stream without callback and use Ctrl+C to stop
    try:
        recorder.start_stream()
    except KeyboardInterrupt:
        recorder.stop_stream()

    # To stream audio and process with a callback function, use the following:
    # recorder = MicrophoneRecorder(device_index=1)  # Specify the device index if needed
    # recorder.start_stream(callback=example_callback)
