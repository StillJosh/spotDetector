# Start with a base SageMaker PyTorch image
FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Copy your training script and other files
ADD src /opt/ml/code/src
COPY requirements.txt .

# Install required packages
RUN pip install -r requirements.txt

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM src/train.py

