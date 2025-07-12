## Environment
The code is tested on Windows11, python 3.11. The following packages are required:
```
numpy==1.5.0
pandas==1.2.3
matplotlib==3.7.1
scikit-learn==0.20.0
seaborn==0.12.2
tensorflow==1.4.0
```

## Dataset
The following database has been preprocessed and can be directly input into the memristor for mapping and training. The collection and preprocessing of raw data are detailed in the Methods section of the article.
- `test_images`:Contains the database for identifying fashion mnist.
- `action-data`:The basic dataset for multimodal tasks containing striking actions is already included. 
- `trajectory_data`:The basic dataset for multimodal tasks containing swing trajectory is already included. 

## Dynamic information recognition stage
This is based on the physical reservoir memoristor model, used for dynamic information recognition to achieve recognition of striking actions and swing trajectory. After installing the environment, use .Run to execute network
- `action-task.py`:The preprocessed striking actions database has been used for training and testing.
- `trajectory_data.py`:Train and test the preprocessed swing trajectory database.
- `fashion_mnist`:Training and testing fashion mnist database.

## Result
Basic results of training
- `action_results`:The training results include striking actions, such as confusion matrix, training process, etc.
- `activation_reports`:The F function includes various swing trajectories.
- `fashion_mnist_results`:Training results containing confusion matrix.