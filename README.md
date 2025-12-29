# sentiment_api


A simple Sentiment Analysis REST API built with PyTorch (LSTM model) and FastAPI.
This API predicts whether a given text expresses Positive or Negative sentiment.


🚀 Features
Text preprocessing (cleaning URLs, mentions, special characters).

Vocabulary encoding with <PAD> and <UNK> tokens.

LSTM-based sentiment classifier implemented in PyTorch.

FastAPI endpoint for easy integration with applications.

JSON-based request/response format.


sentiment-api/
│
├── app.py                 # FastAPI application
├── sentiment_checkpoint.pth  # Saved model + vocab
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── notebooks/             # (Optional) Jupyter notebooks for training

⚙️ Installation
Clone the repository:
git clone https://github.com/adityatiwari102/sentiment-api.git
cd sentiment-api

Install dependencies:
pip install -r requirements.txt

▶️ Running the API
Start the FastAPI server with Uvicorn:

uvicorn app:app --reload

The API will be available at:
http://127.0.0.1:8000

Interactive Swagger docs:
http://127.0.0.1:8000/docs

🧠 Model Details
Architecture: LSTM with embedding + fully connected layer.

Embedding dimension: 128

Hidden dimension: 256

Output dimension: 2 (Positive / Negative)

Training: Preprocessed text dataset with vocabulary encoding.

📌 Requirements
Python 3.8+

PyTorch

FastAPI

Uvicorn

Pydantic

