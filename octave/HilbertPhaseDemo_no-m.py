# % Why broadband phase is not a thing, illustrated by the Hilbert transform
# % Many papers deal with phase relationships between signals collected at different time points or at different locations, and/or different frequencies (cross-frequency coupling). Surprisingly many of these analyses are unlikely to have worked because they measured phase from broadband signals. This walkthrough uses the hilbert transform to show why broad band phase is not a thing and how not taking this seriously yield spurious results that will never ever replicate. 
# % first load exampe data, make sure the data are in the current directory,
# % or in the path


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt, hilbert
import scipy.io as sio



# %% [markdown]
# % first load exampe data, make sure the data are in the current directory,
# % or in the path

# %%
bop_107 = sio.loadmat('bop_107.fl40h1.E1.app1.mat')
outmat = bop_107['outmat']

# %%

tx = sio.loadmat('taxis.mat')
taxis = np.squeeze(tx['taxis'].copy())

# %%
print(outmat.shape)

SampRate = 500
data = outmat[:,:,:] # loads our example data set (129 sensors, 2551 points,  48 trials)
data = np.squeeze(outmat[128,:, 1]).T  # the second trial, sensor Cz. 

plt.figure()
plt.plot(taxis,data)
plt.title ('Our example data')



#--
# %% [markdown]
#% Next filter the signal a bit, but not enough...let us start with 7 to 12 Hz, often used in the literature, 


# %%
alow,blow = butter(4, 12/250)   # 12 Hz lowpass when sampled at 500 Hz
siglow = filtfilt(alow, blow, data)
ahigh,bhigh = butter(2, 7/250, 'high') # 7 Hz highpass (2nd order) when sampled at 500 Hz

sighighlow = filtfilt(ahigh, bhigh, siglow)

plt.figure()
plt.plot(taxis, sighighlow)
plt.title(' signal, filtered 7 to 12 Hz')



#--
# %% [markdown]
#% Now we apply the Hilbert transform. It estimates the local phase time-point by time point and creates a version that is shifted by 90 degrees. it is often used in CFC studies. For example 4 Hz in Canolty et al. Science 2016


# %%

test = hilbert(sighighlow)
plt.figure()
plt.plot(taxis, test.imag)
plt.title('imaginary part of the hilbert transform')
plt.xlim([-1263, -78])
plt.ylim([-0.214, 0.280])
plt.show()

plt.figure()
plt.plot(taxis, np.angle(test))
plt.title('screwed up phase :-)')
plt.show()

# %%
plt.figure()
plt.plot(taxis, np.angle(test))
plt.xlim(-3141, -2698)
plt.ylim([-2.71, 2.63])
plt.title('screwed up phase zoomed in')
plt.show()



#--
# %% [markdown]
# %now do it again, with narrow band bass :) 


# %%
alow,blow = butter(4, 12/250)   # 12 Hz lowpass when sampled at 500 Hz
siglow = filtfilt(alow, blow, data)
ahigh,bhigh = butter(2, 7/250, 'high') # 7 Hz highpass (2nd order) when sampled at 500 Hz

sighighlow = filtfilt(ahigh, bhigh, siglow)

plt.figure()
plt.plot(taxis, sighighlow)
plt.title(' new signal, hilbert analytical signal, and envelope')

#--


# %%
test = hilbert(sighighlow)

plt.figure()

plt.plot(taxis, test.imag)
plt.title('imaginary part of the hilbert transform')

# %%

plt.figure()
plt.plot(taxis, np.abs(test))

plt.figure()
plt.plot(taxis, np.angle(test))

plt.title('still pretty bad phase')




#--
# %% [markdown]
#% now do it again, with narrow band pass 


# %%

alow, blow = butter(6, 10/250)   # lowpass when sampled at 500 Hz
siglow = filtfilt(alow, blow, data)

ahigh, bhigh = butter(6, 10/250, 'high') #  highpass (2nd order) when sampled at 500 Hz 
sighighlow = filtfilt(ahigh, bhigh, siglow)

plt.figure()
plt.plot(taxis, sighighlow)
plt.title(' new signal, hilbert analytical signal, and envelope')


# %%
test = hilbert(sighighlow)

plt.figure()
plt.plot(taxis, test.imag)

plt.figure()
plt.plot(taxis, np.abs(test))

plt.figure()
plt.plot(taxis, np.angle(test))
plt.title('that is smooth phase !')



#--
