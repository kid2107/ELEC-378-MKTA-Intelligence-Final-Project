We have attached the three models that we used for this project; a SVM that that got a max accuracy of 2%, a Random Forest
classifier that got a max accuracy of 40%, and a convolutional neural network that got a max accuracy of 90%.

All of our code is commented for clarity, and all three models use the same functions to implement various aspects of the pipeline
including reading the images and writing the final submission file.

In order to run our code and train the models, you will need the following; a folder, in the same workspace as the code, called
"train_images" which should contain all of the training images; a folder, in the same workspace as the code, called
"test_images" which should contain all of the test images; and a file called "train.csv" which should contain all of the labels
for the training images. Furthermore, you should have the following python libraries - numpy, pandas, matplotlib, PIL, sklearn,
skimage, tensorflow, and warnings. Additionally, the Random Forest classifier and CNN classifier were initally designed on colab,
hence they have the following two lines:
  from google.colab import drive
  drive.mount('/content/drive')
These can be commented out if you are running the file offline on your device.

Finally, you should be aware that the code, by default, writes the submission into a file called "submission.csv", and it will
overwrite any preexisting file with that name in the directory. You can use the 'file=' argument to the 'makeSubmission()'
function (line 144 for the SVM, line 216 for the Random Forest, line 244 for the CNN) to change this by simply inputting a
different string to the 'file=' argument. That string will then be used as the name for the file. You must include the .csv
file extension in the string if you are giving the function a new name for the submission file.
