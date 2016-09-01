**Faint pipeline wisdom**

- Install fftw3
On mac:
```
    brew install fftw
```
On Fedora:
```
    yum install fftw3
```

- Install pyfftw (also see their github page for details) [use the right pip! (the one from anaconda)]:
```
    pip install pyfftw
```

- Clone image_registration repository from https://github.com/dmitryduev/image_registration.git
 I've made it use pyfftw by default, which is significantly faster than the numpy's fft,
 and quite faster than the fftw3 wrapper used in image_registration by default:
```
    git clone https://github.com/dmitryduev/image_registration.git
```

- Install it:
```
    python setup.py install --record files.txt
```

- To remove:
```
    cat files.txt | xargs rm -rf
```
