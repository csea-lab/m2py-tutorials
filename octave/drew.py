# %% [markdown]
#% FFT demo - comparing FFT and dot product
# %% [markdown]
#% Start by making a discrete time vector... 

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

time = np.arange(0, 1, 0.001) # one second of discrete time
time.shape

# %%
time1k1 = np.arange(0, 1.001, 0.001) # one second of discrete time
time1k1.shape

# %% [matlab]
# time = 0.001:0.001:1; % one second of discrete time
#--

# %% [markdown]
#% ... and 4 simple signals, to be added later into 2 testsignals. 

# %%
a=np.sin(2*np.pi*time*10) # 10 Hz
b=np.sin(2*np.pi*time*11) # 11 Hz
c=np.sin(2*np.pi*time*15) # 15 Hz
d=np.cos(2*np.pi*time*15) # 15 Hz


# %% [matlab]
# a = sin(2*pi*time*10); % 10 Hz
# b = sin(2*pi*time*11); % 11 Hz
# c = sin(2*pi*time*15); % 15 Hz
# d = cos(2*pi*time*15); % 15 Hz
#--

# %% [markdown]
#% One test signal is sines only

# %%
testsig_sines = a+b+c


# %% [matlab]
# testsig_sines = a+b+c; 
#--
# %% [markdown]
#% The other is a mix of 2 sines and 1 cosine, same frequencies

# %%
testsig_sincos = a+b+d

plt.plot(time, testsig_sines)
plt.plot(time, testsig_sincos)
plt.show()

# %% [matlab]
# testsig_sincos = a+b+d; 

# plot(time, testsig_sines), 
# hold on 
# plot(time, testsig_sincos)

#--
# %% [markdown]
#% First, the regular discrete Fourier transform on the each data segment, resulting in complex spectra with real and imag components

# %%
fft1 = np.fft.fft(testsig_sines)
fft2 = np.fft.fft(testsig_sincos)

# %% [matlab]
# fft1 = fft(testsig_sines);
# fft2 = fft(testsig_sincos);
#--
# %% [markdown]
#% and now, we do the dot product of integer sines and cosines and the test signals. 

# %%
dot1 = np.zeros(30)
dot2 = np.zeros(30)
cosdot1 = np.zeros(30)
cosdot2 = np.zeros(30)

for x in range(0, 30):
    dot1[x] = np.dot(testsig_sines, np.sin(time*2*np.pi*x))
    dot2[x] = np.dot(testsig_sincos, np.sin(time*2*np.pi*x))
    cosdot1[x] = np.dot(testsig_sines, np.cos(time*2*np.pi*x))
    cosdot2[x] = np.dot(testsig_sincos, np.cos(time*2*np.pi*x))


# %% [matlab]
# for x = 1:30; sindot1(x) = testsig_sines*sin(time*2*pi*x)'; end
# for x = 1:30; sindot2(x) = testsig_sincos*sin(time*2*pi*x)'; end
# for x = 1:30; cosdot1(x) = testsig_sines*cos(time*2*pi*x)'; end
# for x = 1:30; cosdot2(x) = testsig_sincos*cos(time*2*pi*x)'; end

#--
# %% [markdown]
#% Plot the resulting spectra

# %%
x=range(0, 29)
f=range(2, 31)
plt.plot(x, np.abs(fft1[2:31]), 'b')
plt.plot(x, dot1[0:29], 'r')
plt.title('Signal with only sines')
plt.legend(['Fourier amplitude', 'Sine-based dot Product'])
plt.show()


plt.plot(x, np.abs(fft2[2:31]), 'b')
plt.plot(x, dot2[0:29], 'r')
plt.title('Signal with sines and cosine')
plt.legend(['Fourier', 'Sine-based dot Product'])
plt.show()

plt.plot(x, np.angle(fft1[0:29]))
plt.plot(x, np.angle(fft2[0:29]))
plt.show()

# %% [matlab]
# figure
# plot(1:30, abs(fft1(2:31)), 'r')
# hold on
# plot(1:30, dot1(1:30), 'b')
# title('Signal with only sines')
# legend('Fourier amplitude', 'Sine-based dot Product')

# figure
# plot(1:30, abs(fft2(2:31)), 'r')
# hold on
# plot(1:30, dot2(1:30), 'b')
# title('Signal with sines and cosine')
# legend('Fourier', 'Sine-based dot Product')

# figure
# plot(0:30, angle(fft1(1:31)))
# hold on
# plot(0:30, angle(fft2(1:31)))
#--
# %% [markdown]