# ✍️ Handwritten Character Recognition

A CNN-based recognizer for handwritten digits and letters, served through a
Streamlit app with a live drawing canvas. Toggle between digit mode (MNIST)
and letter mode (EMNIST) and get real-time predictions with confidence scores.

## Results

| Model         | Dataset        | Classes | Test Accuracy |
|---------------|----------------|---------|----------------|
| Digit model   | MNIST          | 0-9     | 99.26%         |
| Letter model  | EMNIST Letters | a-z     | 93.07%         |

## Demo

Draw a character on the canvas, and the app predicts it live with a top-3
confidence breakdown.

## Tech Stack

- **TensorFlow / Keras** — CNN model, trained on CPU
- **Streamlit** — web app UI
- **streamlit-drawable-canvas** — live drawing input
- **tensorflow-datasets** — EMNIST loading
- **NumPy / Pillow / SciPy** — image preprocessing

## Project Structure

```
handwriting_app/
├── model.py          # shared CNN architecture
├── train_mnist.py    # trains + saves digit_model.h5
├── train_emnist.py   # trains + saves letter_model.h5
├── app.py             # Streamlit UI
├── requirements.txt
└── models/            # created after training (gitignored)
    ├── digit_model.h5
    └── letter_model.h5
```

## Setup

```bash
git clone https://github.com/chayancsk/handwritten-char-recognition.git
cd handwritten-char-recognition
pipenv install -r requirements.txt
pipenv shell
```

## 1. Train the models

Model weights aren't committed to the repo — train them locally:

```bash
python train_mnist.py      # -> models/digit_model.h5   (~15-20 min on CPU)
python train_emnist.py     # -> models/letter_model.h5  (~40-60 min on CPU)
```

Both scripts use early stopping and typically finish before the full 15
epochs. EMNIST letters is a harder task than MNIST digits — some confusion
between visually similar characters (b/d, i/l, o/0) is expected and normal.

## 2. Run the app

```bash
streamlit run app.py
```

Draw a digit or letter on the canvas, toggle between "Digits" and "Letters"
mode, and see the top-3 predictions with confidence bars.

## How it works

- **Architecture**: 3 convolutional blocks (32→64→128 filters) with
  BatchNorm and MaxPooling, followed by a dense classifier head with
  dropout. Same architecture is reused for both models — only the output
  layer size changes (10 vs 26 classes).
- **Preprocessing**: canvas input is cropped to its bounding box, resized to
  28×28, and re-centered by pixel mass — mirroring how MNIST/EMNIST images
  are framed, which meaningfully improves accuracy on freehand input versus
  a naive resize.
- **CPU-friendly by design**: kept deliberately lightweight (~390K params)
  so both models train in well under an hour without a GPU.

## Roadmap

- **Word/sentence recognition**: extend to a CRNN (CNN feature extractor +
  BiLSTM + CTC loss) to handle full words without needing pre-segmented
  characters.
- **Confusion matrix / error analysis** for the letter model, to target
  which character pairs need more training data or augmentation.

## License

MIT
