#% Illustrating trade-off between time and frequency resolution


# %%
import numpy as np
import matplotlib.pyplot as plt

#--
# %% [markdown]
#% Start by making a discrete time vector... 

# %%

time = np.arange(0, 1, 0.001) # one second of discrete time, sampled at 1000 Hz
faxis = np.arange(0, 500, 1) # the frequency axis, from 0 to Nyquist limit (500) Hz, in 1 Hz steps


# %% [matlab]
clear
time = 0.001:0.001:1; % one second of discrete time
faxis = 0:500; % frequency axis goes from 0 to nyquist in steps of 1

#--
# %% [markdown]
#% ... now make 50 trials of white noise (stochastic) segments, the same noise to be used for all future simulations

# %%
whitesig = np.random.uniform(low=0.0, high=1.0, size=(50,time.shape[0]))-0.5

for trial in range(50):
    plt.plot(time, whitesig[trial,:])
    




# %% [matlab]
whitesig = zeros(50, length(time));
for trial = 1:50
whitesig(trial,:) = rand(1,1000)-.5; % zero centered white noise
end

figure
plot(time, whitesig(1,:))
hold on

#--
# %% [markdown]
#% make bioreaslistic brownian nose signal using cumsum

# %%





# %% [matlab]
brownsig = cumsum(whitesig(1,:));
plot(time, brownsig)

#--
# %% [markdown]
#% now we add an alpha oscillation at the end of the trial, 500 to 900 ms. 

# %%





# %% [matlab]

 alphasig = sin(2*pi*time*10).*4; 
 brownsig(500:900) = brownsig(500:900)  + alphasig(500:900); 
 plot(time, brownsig)
  

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
#% Same, but we do 10 runs with SAWTOOTH instead of sine-shaped alpha and look at the variability across runs after averaging 50 trials 

# %%





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
