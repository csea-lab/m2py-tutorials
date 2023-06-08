#% Demo: Illustrating 1/f and DFT
#% This demo illustrates some of the issues related to the typical 1/f shape of amplitude and power spectra obtained from neural time series. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# %% [matlab]

#--
# %% [markdown]
#% Figure 1. Example of the 1/f shape of an empirical spectrum, with best fitting exponential function shown in red. Note how the power in teh alpha band range (around 10Hz) deviates from the 1/f shape. Figure from Keil et al. https://doi.org/10.1111/psyp.14052
#%  This shape (see figure above) is often modeled as an exponential function with high amplitudes at lower frequencies and lower amplitudes at high frequencies, and its presence in spectra creates some challenges for measuring and comparing spectral power/amplitude in neursocience studies. We will look at how this shape may obscure signals that researchers may be interested in, but also how trial averaging, widely used in neuroscience impacts the effect of 1/f shape on detecting an oscillation. 
#% We start by making a discrete time vector, at which the signal will be sampled. 

# %%

time = np.arange(0, 1, 0.001) # one second of discrete time, sampled at 1000 Hz
faxis = np.arange(0, 500, 1) # the frequency axis, from 0 to Nyquist limit (500) Hz, in 1 Hz steps

# %% [matlab]
# clear
# time = 0.001:0.001:1; % one second of discrete time
# faxis = 0:500; % frequency axis goes from 0 to nyquist in steps of 1

#--
# %% [markdown]
#% ... now we make 50 trials of white noise (stochastic) segments, the same noise to be used for all future simulations

# %%
whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,1000))-0.5
sumspec=np.zeros((50, 1000))

for trial in range(50):
    plt.subplot(2,1,1)
    plt.plot(time, whitesig[trial, :])
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage')
    plt.title('1 second of white noise')

    fftspec = np.fft.fft(whitesig[trial, :]) # calculate DFT
    sumspec[trial, :] = np.abs(fftspec) # save it for later
    
    plt.subplot(2,1,2)
    plt.plot(np.abs(fftspec[1:30]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude spectrum')
    #plt.pause(0.05)
    #plt.show()

plt.figure()
plt.plot(np.mean((sumspec[:, 1:30]), axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum')

# %% [matlab]
# whitesig = rand(50,1000)-.5; % zero centered white noise

# for trial = 1:50
#     subplot(2,1,1), plot(time, whitesig(trial, :)), xlabel('Time (sec)'), ylabel('Voltage'), title('1 second of white noise')
#     fftspec = fft(whitesig(trial, :)'); % calculate DFT
#     sumspec(trial, :) = abs(fftspec); % save it for later
#     subplot(2,1,2), plot(abs(fftspec (2:30))), xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Amplitude spectrum')
#     pause(.2)

# end

# figure
# plot(mean((sumspec(:, 2:30))))
# xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum')

#--
# %% [markdown]
#% now we do it again with the cumulative sum of the white noise, which results in brownian noise

# %%
brownsig = np.zeros((50, 1000))
sumspec = np.zeros((50, 1000))

for trial in range(50):
    brownsig = np.cumsum(whitesig[trial, :])
    plt.subplot(2,1,1)
    plt.plot(time, brownsig)
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage')
    plt.title('1 second of Brownian noise')
    fftspec = np.fft.fft(brownsig) # calculate DFT
    sumspec[trial, :] = np.abs(fftspec) # save it for later 
    plt.subplot(2,1,2)
    plt.plot(np.abs(fftspec[1:30]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude spectrum')
    #plt.pause(0.05)
    #plt.show()
plt.show()

plt.figure()
plt.plot(np.mean((sumspec[:, 1:30]), axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum')



# %% [matlab]
# for trial = 1:50

#     brownsig(trial,:) = cumsum(whitesig(trial, :));
#     subplot(2,1,1), plot(time, brownsig(trial,:)), xlabel('Time (sec)'), ylabel('Voltage'), title('1 second of Brownian noise')
#     fftspec = fft(brownsig(trial,:)); % calculate DFT
#     sumspec(trial, :) = abs(fftspec); % save it for later
#     subplot(2,1,2), plot(abs(fftspec (2:30))), xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Amplitude spectrum')
#     pause(.2)

# end

# figure
# plot(mean((sumspec(:, 2:30))))
# xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum')

#--
# %% [markdown]
#% now we add a variable alpha oscillation to each trial

# %%
whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,1000))-0.5
brownsig = np.zeros((50, 1000))

for trial in range(50):
    brownsig[trial, 300:700] = brownsig[trial, 300:700] #+ np.sin(2*np.pi*time[300:700]*(8+np.random.rand(1)*3))*3
    alphasig= np.sin(2*np.pi*time*(8+np.random.rand(1)*3))*3
    brownsig[trial, 300:700] = brownsig[trial, 300:700] + alphasig[300:700]
    plt.subplot(2,1,1)
    plt.plot(time, brownsig[trial,:])
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage')
    plt.title('1 second of Brownian noise')

    fftspec = np.fft.fft(brownsig[trial,:]) # calculate DFT
    sumspec[trial, :] = np.abs(fftspec) # save it for later 
    
    plt.subplot(2,1,2)
    plt.plot(np.abs(fftspec[1:30]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude spectrum')
    #plt.pause(0.05)
    #plt.show()


plt.figure()    
plt.plot(np.mean((sumspec[:, 1:30]), axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum')



# %% [matlab]

# for trial = 1:50
#     brownsig(trial,:) = cumsum(whitesig(trial, :));
#     alphasig = sin(2*pi*time*(8+rand(1,1).*3)).*3;
#     brownsig(trial, 300:700) = brownsig(trial, 300:700) + alphasig(300:700); 
#     subplot(2,1,1), plot(time, brownsig(trial,:)), xlabel('Time (sec)'), ylabel('Voltage'), title('1 second of Brownian noise')
#     fftspec = fft(brownsig(trial,:)); % calculate DFT
#     sumspec(trial, :) = abs(fftspec); % save it for later
#     subplot(2,1,2), plot(abs(fftspec (2:30))), xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Amplitude spectrum')
#     pause(.2)

# end

# figure
# plot(mean((sumspec(:, 2:30))))
# xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum')

#--
# %% [markdown]
#% Same, but we do 10 runs and look at the variability across runs after averaging 50 trials 

# %%
whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,1000))-0.5
brownsig = np.zeros((50, 1000))
sumspec = np.zeros((50, 1000))
sumspecsine = np.zeros((50, 1000))

for run in range(10):
   for trial in range(50):
    brownsig[trial,:] = np.cumsum(whitesig[trial, :])
    alphasig = np.sin(2*np.pi*time*(8+np.random.rand(1)*3))*3
    brownsig[trial, 300:700] =+ alphasig[300:700]
    fftspec = np.fft.fft(brownsig[trial,:]) # calculate DFT
    sumspecsine[trial, :] = np.abs(fftspec) # save it for later

plt.plot(np.mean(sumspecsine[:, 2:30], axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum with Sine-shaped alpha')



# %% [matlab]
figure, hold on

for run = 1:10
  for trial = 1:50
    brownsig(trial,:) = cumsum(whitesig(trial, :));
    alphasig = sin(2*pi*time*(8+rand(1,1).*3)).*3;
    brownsig(trial, 300:700) = brownsig(trial, 300:700) + alphasig(300:700); 
    fftspec = fft(brownsig(trial,:)); % calculate DFT
    sumspecsine(trial, :) = abs(fftspec); % save it for later
  end

  plot(mean((sumspecsine(:, 2:30))))
  xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum with Sine-shaped alpha')
end
hold off

#--
# %% [markdown]
#% Same, but we do 10 runs with SAWTOOTH instead of sine-shaped akpha and look at the variability across runs after averaging 50 trials 

# %%

whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,1000))-0.5
brownsig = np.zeros((50, 1000))
sumspec = np.zeros((50, 1000))
sumspecsine = np.zeros((50, 1000))

for run in range(10):
   for trial in range(50):
    brownsig[trial,:] = np.cumsum(whitesig[trial, :])
    alphasig = sig.sawtooth(2*np.pi*time*(8+np.random.rand(1)*3))*3
    brownsig[trial, 300:700] =+ alphasig[300:700]
    fftspec = np.fft.fft(brownsig[trial,:]) # calculate DFT
    sumspecsine[trial, :] = np.abs(fftspec) # save it for later

plt.plot(np.mean(sumspecsine[:, 2:30], axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum with Sine-shaped alpha')



# %% [matlab]
figure, hold on

for run = 1:10
  for trial = 1:50
    brownsig(trial,:) = cumsum(whitesig(trial, :));
    alphasig = sawtooth(2*pi*time*(8+rand(1,1).*3)).*3;
    brownsig(trial, 300:700) = brownsig(trial, 300:700) + alphasig(300:700); 
    fftspec = fft(brownsig(trial,:)); % calculate DFT
    sumspec(trial, :) = abs(fftspec); % save it for later
  end

  plot(mean((sumspec(:, 2:30))))
  xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum with non-sine-shaped alpha')
end
hold off

figure
  plot(mean((sumspecsine(:, 2:30))))
  hold on 
  plot(mean((sumspec(:, 2:30))))
  legend('Sine-shaped', 'Sawtooth-shaped')


#--
# %% [markdown]
#% Let's make a dent into the 1/f shape

# %%
whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,1000))-0.5
brownsig = np.zeros((50, 1000))
sumspec = np.zeros((50, 1000))
sumspecsine = np.zeros((50, 1000))

for trial in range(50):
    brownsig[trial,:] = np.cumsum(whitesig[trial, :])-np.mean(np.cumsum(whitesig[trial, :]))
    fftspec = np.fft.fft(brownsig[trial,:]) # calculate DFT
    sumspec[trial, :] = np.abs(fftspec) # save it for later

plt.plot(faxis[2:30],np.mean(sumspec[:, 2:30], axis=0))

for trial in range(50):
    brownsig[trial,:] = np.cumsum(whitesig[trial, :]*2)-np.mean(np.cumsum(whitesig[trial, :]*2))
    #alphasig = sig.sawtooth(2*np.pi*time*(8+np.random.rand(1)*3))*3
    #brownsig[trial, 300:700] =+ alphasig[300:700]
    fftspec = np.fft.fft(brownsig[trial,:]) # calculate DFT
    sumspecsine[trial, :] = np.abs(fftspec) # save it for later

plt.plot(faxis[2:30],np.mean(sumspec[:, 2:30], axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Averaged amplitude spectrum')



# %% [matlab]
for trial = 1:50

    brownsig(trial,:) = cumsum(whitesig(trial, :))-mean(cumsum(whitesig(trial, :)));
    %subplot(2,1,1), plot(time, brownsig(trial,:)), xlabel('Time (sec)'), ylabel('Voltage'), title('1 second of Brownian noise')
    fftspec = fft(brownsig(trial,:)); % calculate DFT
    sumspec(trial, :) = abs(fftspec); % save it for later
    %subplot(2,1,2), plot(abs(fftspec (2:30))), xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Amplitude spectrum')

end

figure
plot(faxis(2:30), mean((sumspec(:, 2:30))))
xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum')
hold on

 % change the gain
for trial = 1:50

    brownsig(trial,:) = cumsum(whitesig(trial, :).*2)-mean(cumsum(whitesig(trial, :).*2));
    %subplot(2,1,1), plot(time, brownsig(trial,:)), xlabel('Time (sec)'), ylabel('Voltage'), title('1 second of Brownian noise')
    fftspec = fft(brownsig(trial,:)); % calculate DFT
    sumspec(trial, :) = abs(fftspec); % save it for later
    %subplot(2,1,2), plot(abs(fftspec (2:30))), xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Amplitude spectrum')

end

plot(faxis(2:30),mean((sumspec(:, 2:30))))
xlabel('Frequency (Hz)'),ylabel('Amplitude'),title('Averaged amplitude spectrum')
hold on
#--
