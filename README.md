

###Prerequisites
 - The code is mostly in MATLAB, except the workhorse of `fhog.m`, which is written in C and comes from Piotr Dollar toolbox http://vision.ucsd.edu/~pdollar/toolbox
 - To use the webcam mode (`runTracker_webcam`), install MATLAB's webcam support from http://mathworks.com/hardware-support/matlab-webcam.html

###Modes
* `runTracker(sequence, start_frame)` runs the tracker on `sequence` from `start_frame` onwards.
* `runTracker_webcam` starts an interactive webcam demo.
* `runTracker_VOT` and `run_Staple` run the tracker within the benchmarks VOT and OTB respectively.


