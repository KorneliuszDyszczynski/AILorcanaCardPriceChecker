#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import webbrowser
import torch
import requests
from bs4 import BeautifulSoup
from PIL import Image
import re
from CardProcessor import process_card_image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

class LorcanaCardParser:
    """Class to parse card data strings and fetch information from URLs."""

    def __init__(self):
        # Regex pattern to enforce a single-digit series number
        self.pattern = re.compile(r'(\d+)(?:/|(?=\d{3}[A-Za-z]))(\d{3})[-\.]?([A-Za-z]+)[-\.]?[\d\.-]*?(\d)$')
        
        # Dictionary to match Lorcana card set numbers to their names
        self.lorcana_card_sets = {
            1: "The First Chapter",
            2: "Rise of the Floodborn",
            3: "Into the Inklands",
            4: "Ursula's Return",
            5: "Shimmering Skies",
            6: "Azurite Sea"
        }

    def parse_data_string(self, data_string):
        """Parse data string to extract card information."""
        match = self.pattern.search(data_string)
        if not match:
            return None

        card_number = match.group(1)
        number_of_cards = match.group(2) if match.group(2) else "204"
        language = match.group(3)
        series_number = int(match.group(4))
        
        set_name = self.lorcana_card_sets.get(series_number, "Unknown Set").replace(" - ", "-").replace(" ", "-").replace("'", "")
        return {
            "card_number": card_number,
            "number_of_cards": number_of_cards,
            "language": language,
            "series_number": series_number,
            "set_name": set_name
        }

    def build_card_url(self, card_info):
            """Constructs the URL for card information."""
            base_url = 'https://lorcanagrimoire.com/card/'
            return f"{base_url}{card_info['card_number']}-{card_info['number_of_cards']}-{card_info['language']}-{card_info['series_number']}"
        
    def get_card_name_and_url(self, url):
        """Fetch the card name from Lorcana Grimoire given a card URL."""
        response = requests.get(url)
        if response.status_code != 200:
            return None, None  # Return None for name and market URL if request fails

        soup = BeautifulSoup(response.text, 'html.parser')
        name_tag = soup.find('h1', class_='card-title grmr-card-h1')
        if name_tag:
            name = name_tag.text.strip().splitlines()[0]
            market_url = "https://lorcanaplayer.com/card/" + name.replace(" - ", "-").replace(" ", "-").replace("'", "").lower() + "/"
            return name, market_url
        return None, None

    def get_market_price(self, market_url):
        """Fetch the price of the card from Lorcana Player given a market URL."""
        response = requests.get(market_url)
        if response.status_code != 200:
            return "Price not found", "Market link not found"

        soup = BeautifulSoup(response.text, 'html.parser')
        price_tag = soup.find("a", {"class": "button price-btn"})
        if price_tag:
            price = price_tag.find("span", {"class": "btn-price"}).text.strip()
            link = price_tag.get('href', 'Link not found.')
            return price, link
        return "Price not found", "Market link not found"

    def fetch_card_data(self, card_info):
        """Fetches card name and price data by calling individual scraping methods."""
        url = self.build_card_url(card_info)
        name, market_url = self.get_card_name_and_url(url)
        if not name:
            return {"Name": "Not found", "price": "N/A", "url": "Link not found."}

        # Fetch the market price if name and market URL are successfully retrieved
        price, price_link = self.get_market_price(market_url)
        return {"Name": name, "price": price, "url": price_link}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images_path', type=str, help='Path to directory of images')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # Load and prepare model for inference
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    card_parser = LorcanaCardParser()

    # Process each image in the specified directory
    image_files = sorted([f for f in os.listdir(args.images_path) if f.endswith(('png', 'jpeg', 'jpg'))])
    for image_file in image_files:
        file_path = os.path.join(args.images_path, image_file)

        # Prepare image for model input
        image = Image.fromarray(process_card_image(file_path))
        image = img_transform(image).unsqueeze(0).to(args.device)

        # Model inference
        prediction = model(image).softmax(-1)
        pred_text, _ = model.tokenizer.decode(prediction)
        
        # Parse and fetch card data
        card_info = card_parser.parse_data_string(pred_text[0])
        if card_info:
            card_data = card_parser.fetch_card_data(card_info)
            print(f"{image_file}: {card_data}")
        else:
            print(f"{image_file}: No valid card information found in prediction")

if __name__ == '__main__':
    main()
