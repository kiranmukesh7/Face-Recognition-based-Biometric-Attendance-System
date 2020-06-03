# Face-Recognition-based-Biometric-Attendance-System
Face Recognition by Eigenface method with the trained Feed Forward Neural Network and other classifiers applied to biometric attendance system functional on static-images.

# Regarding files and folders:

1. The jupyter notebook "Face_Detection_Recognition.ipynb" is a building code for the complete project. It contains the complete set of codes which were used for building the final functional code. It contains the hand-coded versions of the neural network(s), face alignment code(though, the method applied did not work on the given dataset) and other experiments performed.

2. Pre-trained models for each dataset have been saved in the respective folders bearing the codes and files of each dataset.

3. Report present in this folder contains complete description of all the experiments conducted and the resources used, steps in pipeline in the face recognition system. It also contains a pseudo code of the submitted code. 

4. The folder "Attendance" contains the attendance updating codes of two type:
	a) One which takes static input images with single subject face (update_attendance_LFW.py or update_attendance_LFW_FFNN.py)
	b) Another which takes static input images with multiple subject faces (update_attendance_Washington.py or update_attendance_Washington_FFNN.py)

5. Type the following command on the terminal after feeding in or modifying the paths of directories for input images, folder for saving name tagged images:

$ python3 <update_attendance_file.py> <dd.mm.yyyy> <1/0>

where, input 1 if one wants to save the output name tagged images in separate folder. In case the date of updation of the attendance is the same as that of execution of the above command, then there is no need to mention the date. 

For further details, kindly go through the report.
