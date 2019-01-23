# lg-depthmap
Tensorflow single image convolutional depth mapping for use with [Middlebury 2014 dataset](http://vision.middlebury.edu/stereo/data/scenes2014/).
## Building numpy datasets
To build the dataset on all max-size squares in all images run
```
python build_data_np.py
```  
This expects data to be located at `mid/data/<scene>-<quality>/` in the format and file pattern expressed in the [Middlebury 2014 stereo dataset](http://vision.middlebury.edu/stereo/data/scenes2014/).
### Sampling/Validation
A folder named validation can be polulated with images to be fed into the network. To convert the test files to np arrays and populate `validation_samples/` with the images and the predicted depthmaps run:
```
python build_validation.py
python sample.py
```
## Dependencies (POTENTIALLY INCOMPLETE)
* numpy
* tensorflow
* PIL
* scipy
* scikit
* matplotlib