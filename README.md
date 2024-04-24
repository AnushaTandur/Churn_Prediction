## End to End Machine Learning Project


### Step 1: Create a new environment

```
conda create -p venv python==3.11

conda activate venv/
```
### Step 2: Create a .gitignore file

```
create the file by right click and include the venv in it
```

### Step 3: Create a requirements.txt file 
```
pip install -r requirements.txt
```

### Step 4: Create a setup.py file 
```
This is to install the entire project as a package.
```

### Step5: Create a folder `src` 
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The scr folder will include another folder with name components will be created. Include __init__.py also 
```
#### Step 5.1 Create a folder `components`

```
Include data_ingestion, data_transformation, model trainer, and __init_.py. 
#### Step 5.2 Create a folder called `pipeline`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
``` 

### Step 6: Create a folder  `notebooks` 
```
Create a folder called data and include the dataset. 
```