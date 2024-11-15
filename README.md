# AI Lorcana Price Checker


https://github.com/user-attachments/assets/011e11a1-77d1-42a8-af18-bec9b63ba389


## Overview

**AI_Lorcana_Price_Checker**  is a fun side project designed to process images of Lorcana cards, extract text, and retrieve additional card information such as names and prices. The system leverages deep learning for text recognition, along with web scraping to gather card details from external sources.

## Features

- **Scene Text Recognition**: The model utilizes **Clip4str** for high-accuracy scene text recognition to extract text from images of Lorcana cards.
- **Card Parsing**: Extracts information such as card number, set, language, and series number.
- **Web Scraping**: Fetches additional card details (like name and price) from external websites such as Lorcana Grimoire and Lorcana Player.
- **Image Preprocessing**: Processes card images with OpenCV to identify and isolate the text region, leveraging contour detection and perspective transformation for optimal text extraction.

### Dependency

Requires `Python >= 3.8` 

```
conda create --name clip4str python=3.8.5
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
pip install -r requirements.txt 
```

Download pre-trained model from official [CLIP4STR](https://github.com/VamosC/CLIP4STR/tree/main) repo.

## Usage
**Step 1**: Prepare your images
Ensure that your card images are placed in a directory. The script expects the images to be in .png, .jpeg, or .jpg format.

**Step 2**: Run the script
The main script processes the images in a given directory, performs text recognition using Clip4str, and extracts the card data. You can run the script with the following command:

```
python3 recognize_cards.py  <path_to_model_checkpoint> --images_path <path_to_images_directory> 
```
--checkpoint: Path to the model checkpoint or the model ID for a pre-trained model.
--images_path: Path to the directory containing the images of Lorcana cards.

**Check the output**
The script will process each image in the directory, predict the card text, parse the data, and fetch card details. For each image, the output will show the card's name, price, and URL.

Example output:
```
card1.jpg: {'Name': 'Mickey Mouse, The Brave', 'price': '$9.99', 'url': 'https://lorcanaplayer.com/card/mickey-mouse-the-brave/'}
card2.jpg: {'Name': 'Elsa, Snow Queen', 'price': '$4.50', 'url': 'https://lorcanaplayer.com/card/elsa-snow-queen/'}
 ```

# How It Works

The process is broken down into the following steps:

## 1. Image Preprocessing
The image is first preprocessed to identify and isolate the card text region. This is done by:

- Converting the image to grayscale.
- Thresholding and finding contours to locate the card.
- Applying a perspective transformation to get a top-down view of the card.

The **CardProcessor** class (in `cropCard.py`) handles this preprocessing.

## 2. Text Recognition
Once the card image is processed, **Clip4str** is used for text recognition. This model extracts text from the card image, including its number, set, and language.

## 3. Card Parsing
The extracted text is parsed by the **LorcanaCardParser** class (in `recognize_cards.py`). It identifies key card information like card number, series, and set. 

## 4. Web Scraping
Finally, the program uses web scraping to gather:

- The card’s name from **Lorcana Grimoire**.
- The card’s market price from **Lorcana Player**.

## Future Plans

The next phase for AI_Lorcana_Price_Checker is to develop it into a real-time web service integrated with a user-friendly mobile app. Through the app, users will be able to stream images of their Lorcana cards via their camera. The backend will process these streams, retrieve the card details and prices, and instantly display the information in the app, potentially using a messaging service like RabbitMQ for efficient communication.









