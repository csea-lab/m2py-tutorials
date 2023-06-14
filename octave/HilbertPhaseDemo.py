# % Why broadband phase is not a thing, illustrated by the Hilbert transform
# % Many papers deal with phase relationships between signals collected at different time points or at different locations, and/or different frequencies (cross-frequency coupling). Surprisingly many of these analyses are unlikely to have worked because they measured phase from broadband signals. This walkthrough uses the hilbert transform to show why broad band phase is not a thing and how not taking this seriously yield spurious results that will never ever replicate. 
# % first load exampe data, make sure the data are in the current directory,
# % or in the path


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt, hilbert

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

# %%

# Load raw data
data_path = mne.datasets.ssvep.data_path()
bids_fname = (
    data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)

raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
raw.info["line_freq"] = 50.0

# Set montage
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

SampRate = 500 # Sampling rate in Hz
raw_rs=raw.copy().resample(sfreq=500, verbose=False)

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

# %%
epochs.info['ch_names'][13]


# %%

 #   ----- ## NEED Cz
#s20_data=epo1[0,29] # 29th sensor is Oz
s13_data=epo1[0,13]
print(s13_data.shape)

#epochs.pick_channels(['Oz'])
#mne.pick_channels(epochs.info['ch_names'], ['Oz'])
plt.plot(tme,s13_data)
plt.show()

# %%
s13_data_cut=s13_data[:2551]
taxis=tme[:2551]
#np.shape(s20_data_ct)



# %%
s13_data_cut.shape

SampRate = 500
data = s13_data_cut.copy()

plt.plot(taxis,data)
plt.show()


# %% [matlab]
# clear 
# load('bop_107.fl40h1.E1.app2.mat') % loads our example data set (129 sensors, 2551 points,  48 trials)
# size(outmat)
# SampRate = 500; 
# data = squeeze(outmat(129, :, 2))';  % the second trial, sensor Cz. 
# load taxis.mat
# plot(taxis, data)


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


# %% [matlab]
# [alow,blow] = butter(4, 12/250);   % 12 Hz lowpass when sampled at 500 Hz

# siglow = filtfilt(alow, blow, data);

# [ahigh,bhigh] = butter(2, 7/250, 'high')% 7 Hz highpass (2nd order) when sampled at 500 Hz

# sighighlow = filtfilt(ahigh, bhigh, siglow);

# figure, plot(taxis, sighighlow), title(' signal, filtered 7 to 12 Hz')

#--
# %% [markdown]
#% Now we apply the Hilbert transform. It estimates the local phase time-point by time point and creates a version that is shifted by 90 degrees. it is often used in CFC studies. For example 4 Hz in Canolty et al. Science 2016


# %%

test = hilbert(sighighlow)
plt.figure()
plt.plot(taxis, test.imag)
plt.title('imaginary part of the hilbert transform')
#plt.xlim([-1263, -78])
#plt.ylim([-0.214, 0.280])
plt.show()

plt.figure()
plt.plot(taxis, np.angle(test))
plt.title('screwed up phase :-)')
plt.show()

# %%
plt.figure()
plt.plot(taxis, np.angle(test))
#plt.xlim(-3141, -2698)
plt.xlim(2.0, 2.5)
# plt.ylim([-2.71, 2.63])
plt.title('screwed up phase zoomed in')
plt.show()


# %% [matlab]

# test = hilbert(sighighlow);

# figure, hold on, plot(taxis, imag(test))

# plot(taxis, abs(test)), hold off

# figure, plot(taxis, abs(test))
# xlim([-1263 -78])
# ylim([-0.214 0.280])

# figure, plot(taxis, angle(test)), title('screwed up phase :-)')

# figure, plot(taxis, angle(test)), title('screwed up phase zoomed in')
# xlim([-3141 -2698])
# ylim([-2.71 2.63])



#--
# %% [markdown]
%now do it again, with narrow band bass :) 


# %%
alow,blow = butter(4, 12/250)   # 12 Hz lowpass when sampled at 500 Hz
siglow = filtfilt(alow, blow, data)
ahigh,bhigh = butter(2, 7/250, 'high') # 7 Hz highpass (2nd order) when sampled at 500 Hz

sighighlow = filtfilt(ahigh, bhigh, siglow)

plt.figure()
plt.plot(taxis, sighighlow)
plt.title(' new signal, hilbert analytical signal, and envelope')



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



# %%


# %% [matlab]

# [alow,blow] = butter(6, 10.5/250);   % 12 Hz lowpass when sampled at 500 Hz

# siglow = filtfilt(alow, blow, data);

# [ahigh,bhigh] = butter(4, 9.5/250, 'high')% 7 Hz highpass (2nd order) when sampled at 500 Hz

# sighighlow = filtfilt(ahigh, bhigh, siglow);

# figure(3)

# plot(taxis, sighighlow), title(' new signal, hilbert analytical signal, and envelope')

# test = hilbert(sighighlow);

# hold on, plot(taxis, imag(test))

# plot(taxis, abs(test)), hold off

# figure(4)

# plot(taxis, angle(test)), title('still pretty bad phase')

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


# %% [matlab]

# [alow,blow] = butter(6, 10/250);   % lowpass when sampled at 500 Hz

# siglow = filtfilt(alow, blow, data);

# [ahigh,bhigh] = butter(6, 10/250, 'high')%  highpass (2nd order) when sampled at 500 Hz

# sighighlow = filtfilt(ahigh, bhigh, siglow);

# figure(3)

# plot(taxis, sighighlow), title(' new signal, hilbert analytical signal, and envelope')

# test = hilbert(sighighlow);

# hold on, plot(taxis, imag(test))

# plot(taxis, abs(test)), hold off

# figure(4)

# plot(taxis, angle(test)), title('that is smooth phase !')

#--
