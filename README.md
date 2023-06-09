# m2py-tutorials
Matlab live scripts translated to python with minimal dependencies:

Where to start using these scripts:

- [NIH OER: General Information to install with Anaconda](https://www.nihlibrary.nih.gov/resources/tools/anaconda-and-python)
- [Python for Matlab-users: Anaconda installation of MNE-Python with Spyder IDE](https://mne.tools/stable/install/installers.html)

## Matlab to Python Tutorial Inventory:

### Mathematical foundations: Dot product, Convolution, Discrete Fourier Transform

- [dotproduct.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/dotproduct.ipynb)
- [convolution.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/convolution.ipynb)
- [convolution_complex.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/convolution_complex.ipynb)
- [introto_DFT.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/introto_DFT.ipynb)
- [Fourier_demo.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/Fourier_demo.ipynb)
- [filters.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/filters.ipynb)

### Hilbert Transform, Wavelets, and more

- [HilbertPhaseDemo.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/HilbertPhaseDemo.ipynb)
- [hilbertavgdemo.ipynb](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/hilbertavgdemo.ipynb)
- [introto_FourierUncertainty_no-m.py](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/introto_FourierUncertainty.ipynb)
- [Introto_Oneoverf_no-m.py](https://github.com/csea-lab/m2py-tutorials/blob/main/notebooks/Introto_Oneoverf.ipynb)

In `octave/` dir to be fully completed :
- octave/wavelet_walkthru 
- octave/Crossfreq_coupling
- octave/MNE_SourceEstimation
- octave/GrangerCausality
- octave/MNE_SourceEstimation

## Extensive Guide to using Scipy and Numpy (especially to translate Matlab to Python)
[Scipy Lecture Notes: v 2022.1 ](http://scipy-lectures.org)  
- Tutorials on the scientific Python ecosystem: a quick introduction to central tools and techniques. 
- The different chapters each correspond to a 1 to 2 hours course with increasing level of expertise, from beginner to expert.

### Matplotlib documentation: 

- [example of colored noise generation and inset plots](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_demo.html)

### Decorators:
[Scipy Lecture Notes: Advanced Python - Decorators](http://scipy-lectures.org/advanced/advanced_python/index.html#decorators)

Potential applications:
Use function decorators to curry normal requred information and variables.  For example, in `de_convolution.py`:

    ~@functools.partial(np.convolve, mode='same')
    def convolve(sig, kernel):
        return np.convolve(sig, kernel, mode='same')~

- `@np.vectorize` to vectorize functions
- `@np.fft.fftshift` to shift the zero-frequency component to the center of the spectrum
- `@scipy.signal.nobs` to get the number of observations
- `@scipy.signal.resample` to resample a signal
- normalize data with `@sklearn.preprocessing.scale`
- add additional arguments to functions with `@functools.partial`
- 

Unit tests and assertions:
Pepper in `assert` statements to check for errors.  For example, in `de_convolution.py`:

    assert len(sig) == len(?), \
        'Lengths of the signal and ??? must match to apply ???'

Use `np.testing.assert_allclose` to check for numerical equality.  For example, in `de_convolution.py`:

    np.testing.assert_allclose(np.sum(sig), 0, atol=1e-10), \
        'The signal must have a mean of zero to apply deconvolution'

Use `np.testing.assert_array_equal` to check for array equality.  For example, in `de_convolution.py`:

`np.testing.assert_array_equal(np.shape(sig), np.shape(???))`, \
'The signal and ??? must have the same shape to apply ???'

indicate that defaults are used


## Code isomorphisms in Python relevant to translation from Matlab:

Simulation initiation reference Schemes: Fs,T,L,& t OR ... mathematical notation??? 

Creation of sumulation variables can take many alternative forms.

- random variable and array generation: `randn` -> `np.random.randn`

- array generation: `1` -> `np.array([1])`
- array generation: `:` -> `np.arange`
- array generation: `1:10` -> `np.arange(1, 11)`
- array generation: `1:2:10` -> `np.arange(1, 11, 2)`

- array generation: `linspace` -> `np.linspace` (opt: `endpoint=True/False`)
- array generation: `logspace` -> `np.logspace`
- array generation: `zeros` -> `np.zeros`
- array generation: `ones` -> `np.ones`

- array indexing: `:` -> `:`
- array indexing: `end` -> `-1`
- array indexing: `end-1` -> `-2`


- plotting: `plot` -> `plt.plot`
- plotting: `imagesc` -> `plt.imshow`
- plotting: `axis` -> `plt.axis`
- plotting: `xlabel` -> `plt.xlabel`
- plotting: `ylabel` -> `plt.ylabel`
- plotting: `title` -> `plt.title`
- plotting: `colorbar` -> `plt.colorbar`

