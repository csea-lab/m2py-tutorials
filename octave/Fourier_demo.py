# %% [markdown]
# Fourier walkthrough with visualization
# This live script walks you through the steps of discrete Fourier transform (DFT) and illustrates some of the principles that help or hinder the application of DFT in EEG analysis, inlcuding practical application
# The goal of FFT is to take time domain data (EEG over time) and give you a frequency spectrum (Frequency domain data), i.e. a plot where frequency is on the x and the power or magnitude at each frequency is on the y-axis. 

# %% [markdown]
#First, load the example data

# %%[python]
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig


# %%
import mne

data_path = mne.datasets.ssvep.data_path()
data_path
bids_fname = (
    data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)
raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
raw.info["line_freq"] = 50.0
raw

# %%# Load raw data
data_path = mne.datasets.ssvep.data_path()
bids_fname = (
    data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)

raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
raw.info["line_freq"] = 50.0

# %%
# Set montage
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

SampRate = 500 # Sampling rate in Hz
raw_rs=raw.copy().resample(sfreq=500, verbose=False)



# %%

# Construct epochs
event_id = {"12hz": 255, "15hz": 155}
events, _ = mne.events_from_annotations(raw_rs, verbose=False)
tmin, tmax = -1.0, 20.0  # in s
baseline = None
epochs = mne.Epochs(
    raw_rs,
    events=events,
    event_id=[event_id["12hz"], event_id["15hz"]],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)

# %%

epo1=epochs[0].get_data()
print(epo1.shape)
epochs.info['ch_names']
tme=epochs.times
s20_data=epo1[0,29]
print(s20_data.shape)

#epochs.pick_channels(['Oz'])
#mne.pick_channels(epochs.info['ch_names'], ['Oz'])
plt.plot(tme,s20_data)
plt.show()


# %%
s20_data_cut=s20_data[:2551]
taxis=tme[:2551]
#np.shape(s20_data_ct)
plt.plot(taxis,s20_data_cut)
plt.show()

# %%
# load('bop_107.fl40h1.E1.app1.mat') % loads our example data set (129 sensors, 2551 points, 41 trials)
# load taxis.mat

# outmat.shape
# SampRate = 500

# data = outmat[68,:, 40] # this picks the 68th channel and the 40th trial (out of 41).
# data = squeeze(outmat(68,:, 40))'; % this picks the 68th channel and the 40th trial (out of 41).
# % squeeze gets rid of third dimension tha matlab wants to keep around, even
# % thoug we only have one column vector of data now

data = s20_data_cut.copy()

plt.plot(taxis, data)
plt.title('Our example data')
plt.show()
# figure, plot(taxis, data), title ('Our example data')

# %% [matlab]
# load('bop_107.fl40h1.E1.app1.mat') % loads our example data set (129 sensors, 2551 points, 41 trials)
# load taxis.mat
# size(outmat)
# SampRate = 500; 

# data = squeeze(outmat(68,:, 40))'; % this picks the 68th channel and the 40th trial (out of 41).
# % squeeze gets rid of third dimension tha matlab wants to keep around, even
# % thoug we only have one column vector of data now
# figure, plot(taxis, data), title ('Our example data')


# %% [matlab]
# % First, load the example data
# load('bop_107.fl40h1.E1.app1.mat') % loads our example data set (129 sensors, 2551 points, 41 trials)
# load taxis.mat
# size(outmat)
# SampRate = 500; 

# data = squeeze(outmat(68,:, 40))'; % this picks the 68th channel and the 40th trial (out of 41).
# % squeeze gets rid of third dimension tha matlab wants to keep around, even
# % thoug we only have one column vector of data now
# figure, plot(taxis, data), title ('Our example data')

# %% [markdown]
#Question: why is it interesting to pick a trial from late in the experiment? 
#Now, do a barebone DFT (FFT)

# %%

NFFT= len(data)
fftMat = np.fft.fft(data, NFFT) # in matlab FFT always acros rows within columns
plt.plot(fftMat,'b',linewidth=0.5,alpha=0.5, marker='o',markersize=0.5,markevery=10)
plt.xlabel('real part')
plt.ylabel('imaginary part')
plt.plot([0, fftMat[7]], 'r') # this is one frequency
plt.show()

print('what is the nature of the output of FFT ?')
print(fftMat.shape)
print(fftMat.dtype)
print(fftMat[0:10])

# %% [matlab]
    # NFFT = length(data); 

    # fftMat = fft(data, NFFT);  % in matlab FFT always acros rows within columns
    
    # plot(fftMat, 'o'), xlabel ('real part'), ylabel('imaginary part')
    # hold on
    # plot([0 fftMat(7)], 'r') % this is one frequency
    
    # disp('what is the nature of the output of FFT ?')
    # whos 

# %% [markdown]
# We see that the out put is "complex" so there are two numbers per number in the output, resulting in a 2-D plot. 
# Also, the size of the output (fftMat) is the same as the input (2551 points) ... 

# ![image.png](attachment:image.png)

# %%
RawMag = np.abs(fftMat)
Phase = np.angle(fftMat)

# %% [matlab]
    # RawMag = abs(fftMat); 
    # Phase = angle(fftMat);

# %% [markdown]
# Raw magnitude at each frequency is the modulus (sqrt if the sum of the squares) of the two parts (real and imag). So, the length of the arrow spanned by the real and imag part in the plot above. 
# Phase is the angle between real and imaginary (see above) 
# Let's plot the result, focusing on power/magnitude

# %%
plt.plot(RawMag)
plt.title('Raw barebone Fourier spectrum')
plt.show()


# %% [matlab]
    # figure, plot(RawMag), title('Raw barebone Fourier spectrum')

# %% [markdown]
# Note how this spectrum is symmetrical. This is because of this: 
# ![image.png](attachment:image.png)
# This is called the Nyquist sampling theorem. If you sample a signal digitally, any frequencies greater than half the sampling rate are meaningless repetitions of the actual ones below half the sampling rate. half the sampling rate = Nyquist frequency. 
# if we apply the logic from the slides of Day 1, namely that the frequency resolution (the steps between x-ticks or bins) is 1000/length of the signal, it follows that the frequency steps here are 1000/(2551.*2) = 0.1960. Thus, 

# %%
freq_axis = np.arange(0, SampRate, SampRate/NFFT)
plt.plot(freq_axis, RawMag)
plt.xlabel('Frequency in Hz')
plt.ylabel('Unscaled power (integral)')
plt.title('Power density plot')
plt.show()



# %% [matlab]
# freq_axis  = 0:1000/(length(data).*2):SampRate-1000/(length(data).*2); % sample rate is 500 Hz, so each sample point has 2 ms	
# figure, plot(freq_axis, RawMag), xlabel('Frequency in Hz'), ylabel('Unscaled power (integral)'), title('Power density plot')

# %% [markdown]
# Because of the Nyquist theorem, we may really only interpret the spectrum up onto half of the sampling rate, here in this example up to 250 Hz, the rest of the spectrum (above 250 Hz) is not valid. Below, we take this into account as we calculate the power. But first, power is overall greater when the signal is longer. This is bad when wanting to compare spectra from different studies, or different EEG segments, with different length. Below, we see how less data result in smaller power. 

# %%
# calculate power with shorter segment
fftMatshort = np.fft.fft(data[0:1000]) #just 2 seconds of data
magshort = np.abs(fftMatshort)
#freq_axis_short  = np.arange(0, 1000/(len(data[0:1000])*2), SampRate-0.5)
freq_axis_short=np.fft.fftfreq(len(data[0:1000]), d=1/SampRate)

plt.plot(freq_axis_short, magshort)
plt.xlabel('Frequency in Hz')
plt.ylabel('Unscaled power (integral)')
plt.title('Power density plot: short segment')
plt.legend('note the overall lower power !')
plt.show()

# %%
freq_axis_short=np.fft.fftfreq(len(data[0:1000]), d=1/SampRate)
fix_freq_axis_short=np.fft.fftshift(freq_axis_short)
plt.plot(freq_axis_short, magshort)
plt.xlabel('Frequency in Hz')
plt.ylabel('Unscaled power (integral)')
plt.title('Power density plot: short segment')
plt.legend('note the overall lower power !')
plt.show()



# %% [matlab]
# % calculate power with shorter segment
# fftMatshort = fft(data(1:1000)); %just 2 seconds of data
# magshort = abs(fftMatshort);
# freq_axis_short  = 0:1000/(length(data(1:1000))*2):SampRate-0.5;
# figure, plot(freq_axis_short, magshort), xlabel('Frequency in Hz'), ylabel('Unscaled power (integral)'), title('Power density plot: short segment'), 
# legend('note the overall lower power !')

# %% [markdown]
# Now we consider only the part of the spectrum that is valid and scale the power values, so that the true lower half of the spectrum contains all the power in the signal. We do this for the full signal, not the short one above. 


# %%
Mag = RawMag*2 # multiply by two, then use only lower half
Mag[0] = Mag[0]/2 # the first value (zero Hz) was not symmetric, so no correction here.
Mag = Mag/NFFT # scale by the number of points, so we can compare long and short segments etc
spec = Mag[0:round(NFFT/2)] # take only lower half of the spectrum and plot it
plt.plot(freq_axis[0:1276], spec)
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density)')
plt.title('Normalized power density plot')
plt.show()



# %% [matlab]
# Mag = RawMag.*2; % multiply by two, then use only lower half
# Mag(1) = Mag(1)/2; % the first value (zero Hz) was not symmetric, so no correction here. 
# Mag = Mag/NFFT; % scale by the number of points, so we can compare long and short segments etc
# spec = Mag(1:round(NFFT./2)); % take only lower half of the spectrum and plot it
# figure, plot(freq_axis(1:1276), spec), xlabel('Frequency in Hz'), ylabel('Scaled power (density)'), title('Normalized power density plot')

# %% [markdown]
# Now we have scaled power, but the plot shows irrelevant (filtered) frequency. Let's zoom in 

# %%
plt.subplot(2,1,1)
plt.plot(freq_axis[0:200], spec[0:200])
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density)')
plt.title('Normalized power density plot')
plt.subplot(2,1,2)
plt.plot(freq_axis[0:200], Phase[0:200])
plt.xlabel('Frequency in Hz')
plt.ylabel('Phase in radians (π)')
plt.title('Phase spectrum')
plt.show()


# %% [matlab]
# figure, 
# subplot(2,1,1), plot(freq_axis(1:200), spec(1:200)), xlabel('Frequency in Hz'), ylabel('Scaled power (density, µV^2/Hz)'), title('Normalized power density plot')
# subplot(2,1,2), plot(freq_axis(1:200), Phase(1:200)), xlabel('Frequency in Hz'), ylabel('Phase in radians (π)'), title('Phase spectrum')

# %% [markdown]
### Congrats! this is a valid FFT/DFT. For some applications, this version with no windowing and no averaging within the time segment is best. 
## Part 2: improving the signal-to noise of the estimate. 

# The spectrum on the right is noisy looking and has higher frequency resolution (many x-ticks) than we might want. Instead we woudl like a clear shape and nice signal. In Matlab and most other enviornments, people often use a version of Welch's method, where a window is moved over the signal, DFTs are calculated, and averaged as they are moved. In matlab here is how this works, all in one command: 

# `[Pxx,F] = pwelch(X,WINDOW,NOVERLAP,F,Fs)`

# %%
Pxx,F=sig.welch(data, SampRate, nperseg=500, noverlap=250, nfft=500) # 50% OVERLAP is the default
plt.subplot(2,1,1)
plt.plot(freq_axis[0:200], spec[0:200])
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density, µV^2/Hz)')
plt.title('Normalized power density plot')
plt.subplot(2,1,2)
plt.plot(Pxx[0:40],F[0:40])
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density, µV^2/Hz)')
plt.title('Welch spectrum')
plt.tight_layout()
plt.show()



# %% [matlab]
# [Pxx,F] = pwelch(data,500,[],500,SampRate); % 50% OOVERLAP is the default
# subplot(2,1,1), plot(freq_axis(1:200), spec(1:200)), xlabel('Frequency in Hz'), ylabel('Scaled power (density, µV^2/Hz)'), title('Normalized power density plot')
# subplot(2,1,2), plot(F(1:40), Pxx(1:40)), xlabel('Frequency in Hz'), ylabel('Scaled power (density, µV^2/Hz)'), title('Welch spectrum')

# %% [markdown]
# Note how the welch spectrum has fewer frequency steps because we used many shorter segments within our data segment, did a spectrum, and averaged the resulting spectra together, to obtain a much smmother, high signal-to-noise version. 

# ----

# Finally, windows are also sometimes useful, even without averaging, because they minimize/attenuate on and offset artifacts. The sine and cosine functions that the DFT uses as models for the data are infinite and stationary (see slides), thus they create artfifacts at the beginning and end of the signal. 

# %%
window = np.hanning(2551)
plt.subplot(2,1,1)
plt.plot(window)
plt.title('A Hanning window, one of many window types')

datawindowed= data*window #we multiply this with the data, pointwise
plt.subplot(2,1,2)
plt.plot(datawindowed, 'r')
plt.title('Windowed data (red)')
plt.show()



# %% [matlab]
# window = hanning(2551); 
# figure, 
# subplot(2,1,1), plot(window), title ('A Hanning window, one of many window types')
# % we multiply this with the data, pointwise
# datawindowed = data.* window; 
# subplot(2,1,2), plot(datawindowed, 'r'), title ('Windowed data (red)')

# %% [markdown]
# compare this with the original data

# %%
plt.plot(data, 'b',label='unwindowed')
plt.plot(datawindowed, 'r',label='windowed')
plt.title('Windowed vs. Unwindowed data')
plt.legend()
plt.show()


# %% [matlab]
# figure,  hold on, plot(data, 'b'), plot(datawindowed, 'r'), title ('windowed vs unwindowed'), legend ('unwindowed', 'windowed')

# %% [markdown]
#  Now, we redo the power spectrum for the windowed data. 

# %%
fftwindowed=np.fft.fft(datawindowed)
RawMagwindowed = np.abs(fftwindowed)
Magwindowed = RawMagwindowed*2 # multiply by two, then use only lower half
Magwindowed[0] = Magwindowed[0]/2 # the first value (zero Hz) was not symmetric, so no correction here.
Magwindowed = Magwindowed/NFFT # scale by the number of points, so we can compare long and short segments etc
specwindowed = Magwindowed[0:round(NFFT/2)] # take only lower half of the spectrum and plot it

plt.subplot(2,1,1)
plt.plot(freq_axis[0:200], spec[0:200], 'k')
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density, µV^2/Hz)') 
plt.title('Unwindowed power')
plt.subplot(2,1,2)
plt.plot(freq_axis[0:200], specwindowed[0:200])
plt.xlabel('Frequency in Hz')
plt.ylabel('Scaled power (density, µV^2/Hz)')
plt.title('Windowed power')
plt.tight_layout()
plt.show()



# %% [matlab]
# fftwindowed = fft(datawindowed); 
# RawMagwindowed = abs(fftwindowed);
# Magwindowed = RawMagwindowed.*2; % multiply by two, then use only lower half
# MaMagwindowed(1) = Magwindowed(1)/2; % the first value (zero Hz) was not symmetric, so no correction here. 
# Magwindowed = Magwindowed/NFFT; % scale by the number of points, so we can compare long and short segments etc
# specwindowed = Magwindowed(1:round(NFFT./2)); % take only lower half of the spectrum and plot it

# figure, 
# subplot(2,1,1), plot(freq_axis(1:200), spec(1:200), 'k'), xlabel('Frequency in Hz'), ylabel('Scaled power (density, µV^2/Hz)'), title('Unwindowed power')
# subplot(2,1,2), plot(freq_axis(1:200), specwindowed(1:200), 'k'), xlabel('Frequency in Hz'), ylabel('Scaled power (density, µV^2/Hz)'), title('Windowed power (Hanning)')

# %% [markdown]
# Note the smaller value at 0 (zero) Hz in the windowed version, and its overall smoother look. Sometimes windowing is beneficial because of these effects. Also note the overall smaller/lower power in the windowed version. If no window is necessarym this is an undesirabel effect of using unneeded windows. 
