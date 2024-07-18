# Text-to-Image Stage GAN

This project integrates a Text-to-Image Stage GAN with a Django application. Follow the steps below to set up and run the project on your local machine.

## Prerequisites

Ensure you have the following packages installed on your machine:

- torch
- matplotlib
- io
- base
- numpy
- pandas
- sentence_transformers
- gdown

You can install these packages using pip:

```bash
pip install torch matplotlib io base numpy pandas sentence_transformers gdown
gdown 1-4liMOrMxY2HTy5FPmDr9MjWPjM-P6iy
gdown 1xEr7gLFTfqcaqJ8rkkG0vEFkx-PvJOlc
py manage.py makemigrations
py manage.py migrate
py manage.py runserver
