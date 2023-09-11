{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP4MCsCBQs1PmoZ1pxOrLxy",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Haffizuddin7899/DSAI-LAB/blob/main/app_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install Streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xArUmPxNLokF",
        "outputId": "42b69e33-51e2-4745-8749-c9b470030096"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: Streamlit in /usr/local/lib/python3.10/dist-packages (1.26.0)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from Streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (5.3.1)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (8.1.7)\n",
            "Requirement already satisfied: importlib-metadata<7,>=1.4 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (6.8.0)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (1.23.5)\n",
            "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (23.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow<10,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=6.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (9.0.0)\n",
            "Requirement already satisfied: pympler<2,>=0.9 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (1.0.1)\n",
            "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests<3,>=2.18 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (13.5.2)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (8.2.3)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (4.5.0)\n",
            "Requirement already satisfied: tzlocal<5,>=1.1 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (4.3.1)\n",
            "Requirement already satisfied: validators<1,>=0.2 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (0.22.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (3.1.35)\n",
            "Requirement already satisfied: pydeck<1,>=0.8 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (0.8.0)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (6.3.2)\n",
            "Requirement already satisfied: watchdog>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from Streamlit) (3.0.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->Streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->Streamlit) (3.1.2)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->Streamlit) (4.19.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->Streamlit) (0.12.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->Streamlit) (4.0.10)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<7,>=1.4->Streamlit) (3.16.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->Streamlit) (2023.3.post1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3,>=2.7.3->Streamlit) (1.16.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->Streamlit) (3.2.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->Streamlit) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->Streamlit) (2.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.18->Streamlit) (2023.7.22)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->Streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->Streamlit) (2.16.1)\n",
            "Requirement already satisfied: pytz-deprecation-shim in /usr/local/lib/python3.10/dist-packages (from tzlocal<5,>=1.1->Streamlit) (0.1.0.post0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->Streamlit) (5.0.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->Streamlit) (2.1.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->Streamlit) (23.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->Streamlit) (2023.7.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->Streamlit) (0.30.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->Streamlit) (0.10.2)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->Streamlit) (0.1.2)\n",
            "Requirement already satisfied: tzdata in /usr/local/lib/python3.10/dist-packages (from pytz-deprecation-shim->tzlocal<5,>=1.1->Streamlit) (2023.3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras"
      ],
      "metadata": {
        "id": "F22jpI_KKrWb"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the trained model\n",
        "model = keras.models.load_model('/content/emotion_model.h5')  # Replace with the path to your saved model"
      ],
      "metadata": {
        "id": "r6FemSSoKxA9"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define class labels\n",
        "class_labels = [ \"angry\", \"disgust\", \"fear\", \"happy\", \"sad\", \"surprise\", \"neutral\"]"
      ],
      "metadata": {
        "id": "OIlQZgyeKzu5"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Streamlit app title and description\n",
        "st.title(\"Expression Classification from Facial Images\")\n",
        "st.write(\"Upload an Facial image, and I will classify it as angry, disgust, fear, happy, sad, surprise and neutral \")"
      ],
      "metadata": {
        "id": "KzqtoEgqK3nh"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "z5IodappGyp0"
      },
      "outputs": [],
      "source": [
        "\n",
        "# Image upload widget\n",
        "uploaded_image = st.file_uploader(\"Choose an Facial image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
        "\n",
        "if uploaded_image is not None:\n",
        "    # Display the uploaded image\n",
        "    st.image(uploaded_image, caption=\"Uploaded Image\", use_column_width=True)\n",
        "\n",
        "    # Preprocess and predict the image\n",
        "    image = Image.open(uploaded_image)\n",
        "    image = np.array(image)\n",
        "\n",
        "    # Ensure the image has 3 dimensions (H x W x C)\n",
        "    if len(image.shape) == 2:\n",
        "        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB\n",
        "\n",
        "    image = tf.image.resize(image, (64, 64))  # Resize to the model's input size\n",
        "    image = tf.image.convert_image_dtype(image, tf.float32)\n",
        "    image = np.expand_dims(image, axis=0)  # Add batch dimension\n",
        "\n",
        "    # Make a prediction\n",
        "    prediction = model.predict(image)\n",
        "    predicted_class = class_labels[np.argmax(prediction)]\n",
        "    confidence = np.max(prediction)\n",
        "\n",
        "    # Display the prediction result\n",
        "    st.write(f\"Prediction: {predicted_class} (Confidence: {confidence:.2f})\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "eg724StaR8z3"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}