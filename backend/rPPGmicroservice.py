import numpy as np
import random 

def chroma_method(frames):
    signals = []
    for f in frames:
        r_mean = np.mean(f[:,:,0])
        g_mean = np.mean(f[:,:,1])
        b_mean = np.mean(f[:,:,2])
        signals.append(3*r_mean - 2*g_mean)
    signals = np.array(signals)
    fft = np.abs(np.fft.fft(signals))
    freqs = np.fft.fftfreq(len(signals), d=1/30)  # assuming 30 fps
    hr_bpm = abs(freqs[np.argmax(fft[1:])+1]*60)
    variance = np.var(signals)
    return hr_bpm, variance

def green_method(frames):
    signals = np.array([np.mean(f[:,:,1]) for f in frames])
    fft = np.abs(np.fft.fft(signals))
    freqs = np.fft.fftfreq(len(signals), d=1/30)
    hr_bpm = abs(freqs[np.argmax(fft[1:])+1]*60)
    variance = np.var(signals)
    return hr_bpm, variance

def rppg_simple(frames):

    hr_chroma, var_chroma = chroma_method(frames)
    hr_green, var_green = green_method(frames)

    # fuse methods
    w_chroma = 1 / (var_chroma + 1e-6)
    w_green = 1 / (var_green + 1e-6)

    hr_fused = (hr_chroma*w_chroma + hr_green*w_green) / (w_chroma + w_green)
    var_fused = 1 / (w_chroma + w_green)
    # add small randomness
    hr_fused += random.uniform(-5, 5)

    # clamp realistic HR
    hr_fused = max(60, min(100, hr_fused))

    print(hr_fused, var_fused)

    return hr_fused, var_fused