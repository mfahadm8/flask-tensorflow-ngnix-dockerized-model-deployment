# Image Classification Model Deployment on ARM-based Device
This repository contains code and instructions for deploying an image classification model on an ARM-based device, such as the Jetson Nano, using Flask, TensorFlow, NGINX, and Docker. The model takes an input image and labels it as either scab, rotten, normal, or blotch. Additionally, it provides the relative probability of every other label, and the web portal declares the verdict based on the label with the highest probability. The accuracy of the model is 84 percent.

## Prerequisites
- Jetson Nano or any other ARM-based device with Docker support
- Docker installed on the device- 
- Basic knowledge of Docker, Flask, TensorFlow, and NGINX

## Installation and Setup
- Clone the repository:
```bash
$ git clone https://github.com/mfahadm8/flask-tensorflow-ngnix-dockerized-model-deployment
$ cd flask-tensorflow-ngnix-dockerized-model-deployment
```

- Build and run using Docker Compose
```bash
docker-compose up -d
```

- Access the web portal:
> Open a web browser and go to http://localhost. The deployed image classification model will be accessible through this web portal.

## Directory Structure
The repository has the following directory structure:
```bash
.
├── docker-compose.yml
├── flask_app
│   ├── 84_percent_accuracy.tflite
│   ├── app.py
│   ├── apt.conf
│   ├── Dockerfile
│   ├── Dockerfile.copy
│   ├── requirements.txt
│   ├── static
│   │   ├── blotch.jpeg
│   │   ├── normal1.jpeg
│   │   ├── rotten.jpeg
│   │   └── scab.jpg
│   ├── templates
│   │   └── home.html
│   ├── testing.ipynb
│   └── wsgi.py
├── images.zip
├── nginx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── project.conf
├── opencv
│   └── opencv_contrib
├── README.md
└── run_docker.sh
```

# Model Details
The image classification model provided in this repository is trained using TensorFlow. It accepts input images and labels them into one of four categories: scab, rotten, normal, or blotch. The model also provides the relative probability of each label.

The accuracy of the model is reported to be 84 percent. However, it is worth noting that the model's performance may vary