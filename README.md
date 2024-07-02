# Face Swapping App

This repository contains a face-swapping application using Python and InsightFace, deployed on Streamlit. The application allows users to upload two images (source and target) and swaps the faces from the source image to the target image. The result can be viewed and downloaded.

## Deployment

The application is deployed at: [Face Swap Insight AI](https://face-swap-insight-ai.streamlit.app/)

## Features

- Upload source and target images.
- Detect faces in both images using InsightFace.
- Swap the faces from the source image to the target image.
- Display the result.
- Download the result image.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd face-swap-app
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt


## Usage

1. Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```

2. Open the app in your browser at http://localhost:8501.

## How It Works

1. The user uploads two images: one as the source image and the other as the target image.
2. The InsightFace model detects faces in both images.
3. The faces from the source image are swapped onto the target image using the InsightFace inswapper_128 model.
4. The result is displayed, and an option to download the result image is provided.


## Acknowledgements
InsightFace for the face analysis and swapping models.


## Feel free to adjust the content and structure as needed.
