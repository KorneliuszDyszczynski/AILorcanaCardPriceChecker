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
from CropImage.cropCard import process_card_image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

# Updated regex pattern to enforce a single-digit series number
pattern = re.compile(r'(\d+)(?:/|(?=\d{3}[A-Za-z]))(\d{3})[-\.]?([A-Za-z]+)[-\.]?[\d\.-]*?(\d)$')

# Dictionary to match Lorcana card set numbers to their names
lorcana_card_sets = {
    1: "The First Chapter",
    2: "Rise of the Floodborn",
    3: "Into the Inklands",
    4: "Ursula's Return",
    5: "Shimmering Skies",
    6: "Azurite Sea"
}

# Parsing function
def parse_string(data_string):
    match = pattern.search(data_string)
    if match:
        # Specify the URL you want to open
        url = 'https://lorcanagrimoire.com/card/'
        card_number = match.group(1)
        # Use the 3-digit `numberofcards` if present; otherwise, default to "204"
        number_of_cards = match.group(2) if match.group(2) else "204"
        language = match.group(3)
        series_number = int(match.group(4))
        set_name = lorcana_card_sets.get(series_number, "Unknown Set").replace(" - ", "-").replace(" ", "-").replace("'", "")
        url+= card_number +'-'+ number_of_cards +'-'+ language +'-'+ str(series_number)
        # Send a request to fetch the webpage

        response = requests.get(url)    
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            name = soup.find('h1', class_='card-title grmr-card-h1').text.strip().splitlines()[0]
            market_url = "https://lorcanaplayer.com/card/"+name.replace(" - ", "-").replace(" ", "-").replace("'", "").lower()+"/"
        response = requests.get(market_url)  
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            price_tag = soup.find("a", {"class": "button price-btn"})
            price = price_tag.find("span", {"class": "btn-price"})
            url = price_tag.get('href', 'Link not found.')
        return {
            "Name": name,
            "price": price.text.strip(),
            "url": url,
        }
    else:
        return None  # If no match is found


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images_path', type=str, help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    files = sorted([x for x in os.listdir(args.images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')])

    for fname in files:
        # Load image and prepare for input
        filename = os.path.join(args.images_path, fname)
        image = Image.fromarray(process_card_image(filename))
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {parse_string(pred[0])}')


if __name__ == '__main__':
    main()
