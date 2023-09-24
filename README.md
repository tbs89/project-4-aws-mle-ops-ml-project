```python
```

# Project 4 AWS Machine Learning Engineer Nanodegre

This is the fourth project of the nanodegree **AWS Machine Learning Engineer**. In this project we are going to
set up the right configuration of a machine learning project involving computer resources, production configuration, 
deployment configuration, security, latency and concurrency.

## Notebook Instance Setup

Firstly, we will create a notebook instance from Sagemaker >> *Notebook* >> *Notebook Instances* >> *Create notebook instance*

[](!screenshots/notebook/snap1.png)


For this project we will user for our instance a <code>ml.t3.medium</code> type that we will name "project-4-udacity".

[](!screenshots/notebook/snap2.png)

[](!screenshots/notebook/snap3.png)


## S3 Setup

Next step is to create a new S3 Bucket:

From S3, we will click on Buckets and Create bucket to create our new bucket

[](!screenshots/notebook/snap4.png)

[](!screenshots/notebook/snap5.png)

Then upload the data to our bucket using wget command:

```python
%%capture
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
!aws s3 cp dogImages s3://project-4-udacity-ml/data/ --recursive
```

[](!screenshots/notebook/snap6.png)


After that, we can start our hyperparameter tunning job, setting parameters, estimators and tunners first.
Our estimator will use our <code>hpo.py</code> script as entry point to train the model with the different
hyperparameters values that we set up:

```python

# Hyperparameters: learning rate and batch size.
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}

role = sagemaker.get_execution_role()

objective_metric_name = "Test Loss"
objective_type = "Minimize"

# Estimator
estimator = PyTorch(
    entry_point="hpo.py",
    base_job_name='pytorch_dog_hpo',
    role=role,
    framework_version="1.4.0",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    py_version='py3'
)

#Tuner
tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=2,
    max_parallel_jobs=2,
    objective_type=objective_type
)

```

After that we can set our enviroments variables and star the hyperparameter tunning job:


```python
# Environment variables
os.environ['SM_CHANNEL_TRAINING']='s3://project-4-udacity-ml/data/'  # data in S3 to train the model
os.environ['SM_MODEL_DIR']='s3://project-4-udacity-ml/' # output of the model artifact
os.environ['SM_OUTPUT_DATA_DIR']='s3://project-4-udacity-ml/output/' # location of the output

# hpo.py script access to our environment variables
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()

# Hyperparameter job:
tuner.fit({"training": "s3://project-4-udacity-ml/data/"})
```

Training Jobs Snapshots

[](!screenshots/notebook/oneinstance.png)

[](!screenshots/notebook/fiveinstance.png)


[](!screenshots/notebook/trainingjobs.png)


Deployed Model

[](!screenshots/notebook/oneendpoint.png)


## EC2 Setup

We will first navigate using the console to EC2 and will click on "Launch instaces"

[](!screenshots/ec2/snap1.png)

We give the name *project-4-aws-ml* and select *Deep Learning AMI GPU PyTorch* as AMI:

[](!screenshots/ec2/snap2.png)

[](!screenshots/ec2/snap3.png)


Then we have to select an instance type, in this case we select <code>t3.medium</code> as a reasonable balance
of performance and affordability.

[](!screenshots/ec2/snap4.png)

We will create also a key pair, so we can access, if we needed, from Cloud9:

Key pair: *project4*

[](!screenshots/ec2/keypair.png)

After that, we can launch our EC2 instance:

[](!screenshots/ec2/launching.png)

[](!screenshots/ec2/running.png)


Now we can connect our instance:

[](!screenshots/ec2/connecting1.png)


[](!screenshots/ec2/connecting2.png)


[](!screenshots/ec2/connecting3.png)




We run our first command:

```python
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
```


[](!screenshots/ec2/commands1.png)

and the command to create the directory for the model

```python
mkdir TrainedModels
```

[](!screenshots/ec2/commands2.png)


Next step is to create a blank Python file in order to paste the training code:

```python
vim solution.py
```

We will use the following command to paste the code to the file and press enter:

```python
:set paste
```


After pasting the code, we need to type the following command:

```python
:wq!
```

[](!screenshots/ec2/commands3.png)


Now we are ready to train the model, by typing the following command:

```python
python solution.py
```

[](!screenshots/ec2/commands4.png)
Training model

[](!screenshots/ec2/commands6.png)
Model saved



[](!screenshots/ec2/instancemetrics.png)
Metrics


Comparision SageMaker vs EC2

Training models on EC2 offers extensive customization and control, allowing users to manually manage the entire machine learning workflow, optimizing it based on specific needs. On the other hand, Amazon SageMaker provides a more integrated and automated environment, focusing on streamlining the machine learning process from training to deployment, making it user-friendly and efficient.

EC2:
- High Customization & Control
- Manual Management of Workflow

SageMaker:
- Integrated & Automated Workflow
- User-friendly & Efficient

## Lambda Functions Setup


































