# Image denoising with autoencoder

This is a python script in which a part of the *MNIST* dataset (handwritten digits) is taken and noise is added; then the algorithm eliminates the noise and "reconstructs" the original input.

To run this code you have to install:
* *Keras*
* *CNTK*
* *NumPy*
* *MatPlotLib*
* *Pillow*

You also must change the defautl *Keras* backend to *CNTK*; quick steps to do it:
1. find *Keras config* file in `<$HOME/.keras/keras.json>` (on *Windows* replace `<$HOME>` with `<%USERPROFILE%>`). If there is no file, create it (name it *keras.json*)
1. it must looks like this:
   ```
   {
        "image_data_format": "channels_last",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "cntk"
   }
   ```
Additional notes:   
the program is slow: it took more than one hour to finish the execution on my home PC; the only way to make it faster would be to use *CNTK* with GPU instead of CPU, but to do this the PC must have an Nvidia GPU compatible with CUDA (so I decided to avoid this solution).
