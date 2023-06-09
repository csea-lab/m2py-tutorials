# % Mini filter tutorial 
# % This live script walks through some key elements of digital filtering
# % First, get some example data. Make sure you have a path set in Matlab to the folder example_data, Thanks! 

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt


# %%

#testdata= spatialPattern([1,2551],-2)

#  ???? where is this function defined?



# %% [matlab]
# clear, close all
# testdata = spatialPattern([1,2551],-2); 
# % spatialPattern by Jon Yearsley 1 May 2004 can be used to make 1/f noise
# % which looks like EEG

#--
# %% [markdown]
# % Now, let's make a time axis so that we know when the event happened and plot the signal

# %%

timeaxis= np.arange(-3.6,1.5,0.002) # generate a time axis, in seconds this was sampled at 500 Hz
plt.figure(1)
plt.plot(timeaxis, testdata)
plt.title('Unfiltered EEG signal')


# %% [matlab]
# timeaxis = -3.6:0.002:1.5; % generate a time axis, in seconds this was sampled at 500 Hz
# figure(1), hold on
# plot(timeaxis, testdata)
# title('Unfiltered EEG signal')

#--
# %% [markdown]
# % Now we make a simple Butterworth Filter at 20 Hz lowpass. 
# % We use the Butter function, the inputs are a filter order, and the cutoff, in multiples of the Nyquist frequency, which is simply half of the sample rate (500 Hz)

# %%
cutoff = 20/250

[A,B] = butter(5,cutoff) # this is a 5th order filter, it will have 6 values

# % let's plot the filter coefficients - i.e. the numbers that make up the filter.
plt.subplot(2,1,1)
plt.plot(A)
plt.subplot(2,1,2)
plt.plot(B)


# %% [matlab]
# cutoff = 20/250
# [A, B] = butter(5, cutoff); % this is a 5th order filter, it will have 6 values
#  % let's plot the filter coefficients - i.e. the numbers that make up the filter.  
# figure(2)
# subplot(2,1,1), plot(A), subplot(2,1,2), plot(B)

#--
# %% [markdown]
# % and now, we apply it to the data

# %%

filtered_data = filtfilt(A,B,testdata) # this is a zero phase filter, it will have the same number of values as the input
plt.plot(timeaxis, filtered_data, 'r')
plt.show()

# %% [matlab]

# filtered_data = filtfilt(A, B, testdata);
# figure(1)
# plot(timeaxis, filtered_data, 'r')

#--
# %% [markdown]
# % now, zoom in

# %%

plt.plot(timeaxis, filtered_data, 'r')
plt.xlim([-2.109, -1.691])
plt.ylim([-1.49, 5.9])
plt.show()

# %% [matlab]

# xlim([-2.109 -1.691])
# ylim([-1.49 5.92])
# legend('unfiltered', 'lowpass filtered')

#--
# %% [markdown]
# % now, we add a highpass filter at 2 Hz, same logic

# %%
cutoff = 2/250
[Ah,Bh] = butter(5,cutoff,'high') # this is a 2nd order filter, it will have 3 values
plt.subplot(2,1,1)
plt.plot(Ah)
plt.subplot(2,1,2)
plt.plot(Bh)
plt.show()

# %% [matlab]

# cutoff = 2/250
# [Ah, Bh] = butter(5, cutoff, 'high'); % this is a 2nd order filter, it will have 3 values
# # % let's plot the filter coefficients - i.e. the numbers that make up the filter.  
# figure(3)
# subplot(2,1,1), plot(Ah), subplot(2,1,2), plot(Bh)

#--
# %% [markdown]
# % apply the filter and plot the signal 

# %%

filtered_datah = filtfilt(Ah,Bh,filtered_data)
plt.plot(timeaxis, filtered_datah, 'g')
plt.xlim([-3.5, 1.6])
plt.ylim([-15, 15])
plt.show()



# %% [matlab]

# filtered_datah = filtfilt(Ah, Bh, filtered_data);
# figure(1)
# xlim([-3.5 1.6])
# ylim([-15 15])
# plot(timeaxis, filtered_datah, 'g'), legend('unfiltered', 'lowpass filtered', 'hi & lo filtered filtered')

#--
# %% [markdown]
# % now, zoom in again

# %%
plt.plot(timeaxis, filtered_datah, 'g')
plt.xlim([-2.109 -1.691])
plt.ylim([-1.49 5.92])
plt.show()

# %% [matlab]

# xlim([-2.109 -1.691])
# ylim([-1.49 5.92])
# legend('unfiltered', 'lowpass filtered', 'hi & lo filtered')

#--
# %% [markdown]
# % OK great now how do I report on this and what does this mean? to figure this out, let do a few spectra: 

# %%
FFT_orig = np.abs(np.fft.fft(testdata))
FFT_lp = np.abs(np.fft.fft(filtered_data))
FFT_hp = np.abs(np.fft.fft(filtered_datah))


# %% [matlab]

# FFT_orig = abs(fft(testdata)); 
# FFT_lp =  abs(fft(filtered_data)); 
# FFT_hp = abs(fft(filtered_datah)); 

#--
# %% [markdown]
# % make a frequency axis

# %%

faxis = np.arange(0,250,1000/(len(testdata)*2)) #0.5)

plt.plot(faxis[0:120], FFT_orig[0:120])
plt.plot(faxis[0:120], FFT_lp[0:120])
plt.plot(faxis[0:120], FFT_hp[0:120])
plt.legend(['unfiltered', 'lowpass filtered', 'hi & lo filtered'])


# %% [matlab]

# faxis = 0:1000/(length(testdata).*2):250; 
# figure
# plot(faxis(1:120), FFT_orig(1:120))
# hold on
# plot(faxis(1:120), FFT_lp(1:120))
# plot(faxis(1:120), FFT_hp(1:120))
# legend('unfiltered', 'lowpass filtered', 'hi & lo filtered')

#--
# %% [markdown]
# % now, zoom in again

# %%


plt.figure()
plt.legend(['unfiltered', 'lowpass filtered', 'hi & lo filtered'])
plt.title('zoomed in')
plt.xlim([10, 25])
plt.ylim([-1.49, 50])


# %% [matlab]

# xlim([10 25])
# ylim([-1.49 50])
# legend('unfiltered', 'lowpass filtered', 'hi & lo filtered')

#--
# %% [markdown]
# % and again 

# %%


plt.figure()
plt.legend(['unfiltered', 'lowpass filtered', 'hi & lo filtered'])
plt.title('zoomed in')
plt.xlim([0, 6])
plt.ylim([-1.49, 300])



# %% [matlab]

# xlim([0 6])
# ylim([-1.49 300])
# legend('unfiltered', 'lowpass filtered', 'hi & lo filtered')

#--
