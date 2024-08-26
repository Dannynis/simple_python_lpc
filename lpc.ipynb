import IPython
import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

from IPython.display import display

# Read audio file
input_file = 'p270_001.wav'
audio_signal, fs = librosa.load(input_file,sr=44100)
frame_duration = 2#0.02  # Frame duration in seconds
frame_length = int(fs * frame_duration)  # Number of samples per frame

# Function to synthesize a frame using LPC coefficients
def lpc_synthesis(A, excitation):
    return scipy.signal.lfilter([1], A, excitation)

# Encode audio using LPC
order = 12  # LPC order
frames = [audio_signal[i:i+frame_length] for i in range(0, len(audio_signal), frame_length)]
lpc_coeffs = []
residuals = []

for frame in frames:
    if len(frame) < frame_length:
        frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
    # A, _ = lpc_analysis(frame, order)
    A = librosa.lpc(frame,order=2)
    residual = scipy.signal.lfilter(A, [1], frame)
    lpc_coeffs.append(A)
    residuals.append(residual)

# Decode audio using LPC
synthesized_signal = []

freqs = []
for A, residual in zip(lpc_coeffs, residuals):
    # for i in range(4,11,1):
    #   A = modify_formant_amplitude(A,44100,i,32)
    # freqs.append(get_fromant_freqs_from_lpc(A,fs))
    freqs.append(scipy.signal.freqz(b=A, a=1, fs=fs)[0])
    synthesized_frame = lpc_synthesis(A, residual)
    synthesized_signal.extend(synthesized_frame)

synthesized_signal = np.array(synthesized_signal)

# Write the synthesized audio to a file
output_file = 'output_audio.wav'
sf.write(output_file, synthesized_signal, fs)

# Plot original and synthesized signals
time = np.arange(len(audio_signal)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, audio_signal)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot( synthesized_signal)
plt.title('Synthesized Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

display(IPython.display.Audio(audio_signal, rate=44100))
display(IPython.display.Audio(synthesized_signal, rate=44100))
display(IPython.display.Audio(np.concatenate(residuals), rate=44100))
