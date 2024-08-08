# BachelorGraduateWork
Neural Networks for identifying traffic signs on images uploaded by user

Description of the files:
Training_my_CNN_v1_0_LeNet_5.ipynb - training LeNet-5 model, using only numpy
Recognition_my_CNN_v1_0_LeNet_5.ipynb - recognition and classifying objects on images by trained LeNet-5 model
Training_my_CNN_v1_1_AlexNet.ipynb - training AlexNet model, using only numpy
Traffic_sign_recognition.ipynb - training and recognition traffic signs on images using Keras and TensorFlow
v2_Traffic_sign_training.ipynb - training YOLOv5 model for detection traffic signs on images
v2_Classification_Traffic_Signs.ipynb - testing of trained YOLOv5 model
The folder 'Software_v2.0' contains:
Folder 'icons' - folder with images of traffic signs
class_mapping.json - json file with matches of class and sign name
meaning_mapping.json - json file with matches of class and sign meaning
main.py - software functionality with a convenient interface for recognizing road signs on a user-uploaded image

Warning!!!
1. All files with the .ipynb extension were written in Google Collaboratory. Therefore, for proper operation, the code must be run from this virtual environment;
2. Make sure that all the necessary files are in the correct directories. Upload the required datasets to the required folders;
3. Make sure you have enough RAM in the virtual environment to properly run the model training;
4. There is no best.pt file in the 'Software_v2.0' folder due to the size limitation of downloaded files, which is the result of model training. To run the code, you first need to train the YOLOv5 model and then add the file to the desired directory;
5. The 'icons' folder does not contain a large number of images of road signs, and those that have already been downloaded act as an illustrative example for the operation of the software product;
6. For the same reason as described in point 5, there is no information about road signs in the meaning_mapping.json file.

Notes:
At first, it was assumed that it would be possible to write the neural network code from scratch without using ready-made libraries, which is why the code for the first CNN models LeNet-5 and AlexNet was written. As it turned out, the learning process is very resource- and time-consuming. The transition to the Keras and TensorFlow libraries also did not give a positive result, since only one sign was recognized and all image was taken into account, even where there was no road sign. Thus, it was decided to use a labeled dataset. For the new dataset, it was decided to use a more modern approach to identifying objects in an image - the YOLOv5 model. If desired, the code in main.py can be rewritten to recognize road signs on images in real time.
