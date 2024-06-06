# Air Quality Prediction Model

This project implements a deep learning model using TensorFlow for predicting air quality using Sentinel-5P imagery. The model is built following the SageMaker custom model directory structure and includes a Dockerfile for creating a Docker image.

## Overview

Air quality prediction using Sentinel-5P satellite imagery provides valuable insights into atmospheric composition and pollutant levels. This project leverages deep learning techniques to analyze Sentinel-5P images and make accurate predictions of air quality.

## Usage

### Training

To train the air quality prediction model:

1. Prepare your training dataset, including Sentinel-5P images and corresponding air quality measurements.
2. Upload the training and test datasets to S3.
3. Build a docker image of the model and push it to Amazon ECR.
4. Use SageMaker to train the model.

### Deployment

To deploy the trained model for inference:

1. Create an endpoint on Amazon SageMaker using the trained model.
2. Make predictions by sending HTTP requests to the endpoint.


## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/air-quality-prediction.git
cd air-quality-prediction
```

## Docker Image and Amazon ECR

To build the Docker image and push it to Amazon ECR, follow these steps:

1. Build the Docker image:
   
```bash
docker build -t air-quality-prediction .
```
2. Tag the Docker image with the ECR repository URI:
   
```bash
docker tag air-quality-prediction:latest <ECR_repository_URI>:latest
```
3. Authenticate Docker to your Amazon ECR registry:
   
```bash
aws ecr get-login-password --region <AWS_region> | docker login --username AWS --password-stdin <AWS_account_ID>.dkr.ecr.<AWS_region>.amazonaws.com
```
4. Push the Docker image to Amazon ECR:
   
```bash
docker push <ECR_repository_URI>:latest
```
