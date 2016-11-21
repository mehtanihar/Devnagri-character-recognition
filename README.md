# Devnagri-character-recognition

I have used a multi layer Convolutional Neural Network for classification of 
Devanagri characters using Tensorflow framework


Data

No of classes=104
Format=PNG Image

Command to run the model:
python run_model.py

Results:

92% accuracy obtained on 4 GB NVIDIA GT920M GPU and i7 CPU

Note:You may get less accuracies(2-10%) on some platforms for the same code.
It seems to be a tensorflow bug in handling cache memory. Please try on a server with sufficient cache memory or change permissions that restrict memory by the CPU to a process.


