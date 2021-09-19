# ecoacoustics-speech-detector
This is the public repository for my EGH400 Engineering Honours Thesis project. 
## Background/motivation
More information to come... 
## Usage
More information to come... 
## Speech detection model
Transfer learning using the *[YAMNet model](https://tfhub.dev/google/yamnet/1) was used to train a speech detector. YAMNet is an audio event classification model that predicts 512 different audio events. While one of the categories was speech, this proved to be inaccurate when tested on data recorded by an ecoacoustic device in a noisy, outdoor setting. Transfer learning using the YAMNet embedding outputs was used to develop a speech detection model specifically for urban ecoacoustics. 

More information to come... 
## Training/Validation Data
Inspired by the *[Microsoft Scalable Noisy Speech Dataset](https://github.com/microsoft/MS-SNSD)*, a script was developed to mix speech data with general environmental audio (bird calls, planes, traffic etc.) at various signal-to-noise ratio. The *[Mozilla Common Voice dataset](https://commonvoice.mozilla.org/en)*, a crowd-sourced human speech dataset was used for speech audio. Environmental audio was recorded at the Oxley Creek Common, Samford Ecological Research Facility and a suburban park. 