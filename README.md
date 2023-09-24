# Project 4 AWS Machine Learning Engineer Nanodegre

This is the fourth project of the nanodegree **AWS Machine Learning Engineer**. In this project we are going to
set up the right configuration of a machine learning project involving computer resources, production configuration, 
deployment configuration, security, latency and concurrency.

## Notebook Instance Setup

Firstly, we will create a notebook instance from Sagemaker >> *Notebook* >> *Notebook Instances* >> *Create notebook instance*

[](!./screenshots/notebook/snap1.png)


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
Hyperparameter Tunning Job:

[](!screenshots/notebook/hpojob.png)


[](!screenshots/notebook/hpojob1.png)

Training Jobs: Hyperparameter Tunning

[](!screenshots/notebook/hpojob2.png)


[](!screenshots/notebook/trainingjobshpo.png)

Estimators:

One Instance:

[](!screenshots/notebook/oneinstance.png)



Five Instances:

[](!screenshots/notebook/fiveinstances.png)



Training Jobs: Estimators

[](!screenshots/notebook/trainingjobsestimators.png)



All Training Jobs Completed


[](!screenshots/notebook/alltrainingjobscompleted.png)



Deployed Model

[](!screenshots/notebook/endpoint.png)


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

We will use the following command to paste the code in <code>ec2train1.py</code> to the file and press enter:

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

To create our Lambda function, we will navigate to AWS Lambda in our console and will click on "Create a function"
that uses <code>Python 3.8</code>:

[](!screenshots/lambda/snap1.png)


[](!screenshots/lambda/snap2.png)


Once paste the code we have from the file <code>lambdafunction.py</code> and click on "Deploy":

[](!screenshots/lambda/snap3.png)


Now, if we try to test the Lambda function with the following json body:

```json
{ "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" }
```

We will get an error:


```json
[ERROR] ClientError: An error occurred (AccessDeniedException) when calling the InvokeEndpoint operation: User: arn:aws:sts::064258437334:assumed-role/project-4-lambda-role-o0ai9u1z/project-4-lambda is not authorized to perform: sagemaker:InvokeEndpoint on resource: arn:aws:sagemaker:us-east-2:064258437334:endpoint/pytorch-inference-2023-09-24-18-16-45-923 because no identity-based policy allows the sagemaker:InvokeEndpoint action
```
Basically, the role created when setting our Lambda has not rights to perform the action. To fix this, we will
update the rights adding a new policy to our Lambda function so that it can access Sagemaker. We will do this
through IAM >> Roles >> searching our Lambda in the search bar:


[](!screenshots/lambda/permit1.png)


Now we will add the policy in order to give rights to trigger the test clicking on the role >> Attach policies
and searching for the policy *AmazonSageMakerFullAccess*, selecting it and adding:

[](!screenshots/lambda/permit2.png)

Adding this policy will allow our lambda to access Sagemaker and our Endpoint. Also, by default, our Lambda function
can only handle one request at one. To change this, we will use concurrency in order to give our Lambda the power to 
trigger multiple times at once.

To add concurrency we would need to configure a new version we will click on Actions >> Publish new version:

[](!screenshots/lambda/version.png)


Now we will click on "Provisioned concurrency" and "Edit":

[](!screenshots/lambda/version1.png)

We will set the concurrency to 3, so our Lambda function can handle 3 requests at once:

[](!screenshots/lambda/version2.png)

New concurrency can take a few minutes to be ready:

[](!screenshots/lambda/version3.png)


## Auto-Scaling

Last step for our excersise is to implement Auto-scaling. Auto-scaling referrs to the dynamic adjustment of the instances used with deployed models. This adjustment is based on workload and it's increased or decreased based on this. This is useful in order to ensure that we are charged only for the active instances.

To add Auto-scaling, we will click on Sagemaker >> Endpoints >> Select our Endpoint >> Configure auto scaling

[](!screenshots/lambda/auto1.png)


First, we will increase up to 3 instances as max number of instances:

[](!screenshots/lambda/auto2.png)


Secondly, we will define the scaling policy. This would referr to the number of simultaneous requests that our endpoint would receive in order to trigger the auto-scaling feature and increase the number of instances. This value would be the *Target* and we will set it to 30 requests. Then we have *Scale in cool down* and *Scale our cool down*:

- **Scale in cool down:** Number of seconds that the auto-scaling must wait before increasing the number of instances
- **Scale out cool dow:** Number of seconds that the auto-scaling must wait before decreasing the number of instances


We wil set this to 45 seconds:

[](!screenshots/lambda/auto3.png)

And our endpoint would be ready:

[](!screenshots/lambda/auto4.png)


In this fourth project for the AWS Machine Learning Engineer Nanodegree, the focus has been on configuring and deploying a machine learning project covering aspects like computer resources, security, and concurrency. Initially, a SageMaker notebook instance was set up, followed by creating an S3 bucket for storing project data and subsequently configuring a model with hyperparameters. Training was conducted on both EC2 and SageMaker, emphasizing EC2's customization and SageMaker's user-friendly automation. Lambda functions were implemented to invoke the model endpoint, which involved resolving permission issues and configuring concurrency for handling multiple requests. Finally, auto-scaling was applied to the deployed model to dynamically adjust instances based on the workload, enabling an efficient utilization of resources.





















