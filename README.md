# Hindi Speech Emotion Recognition

This project aims to recognize emotions (e.g., happy, sad, angry, neutral) from speech data in Hindi. The model employs speech signal processing techniques, machine learning, and deep learning to achieve emotion classification. A Streamlit app is also provided for testing the model.

---

## Objective
To create a robust system that classifies emotions in Hindi speech using machine learning and deep learning models.

---

## Project (Pipeline) Flow

1. **Data Ingestion**
    - Identify or prepare a dataset (synthetic or real) for Hindi speech emotion recognition.
    - Organize the dataset and perform initial validation.

2. **Preprocessing**
    - Extract features from speech signals using techniques like MFCCs (Mel-Frequency Cepstral Coefficients).
    - Normalize and clean the data to make it suitable for training.

3. **Model Development**
    - Build multiple machine learning and deep learning models, including CNN, MLP, Random Forest, and KNN, to classify emotions.

4. **Model Evaluation**
    - Evaluate the models' performance using metrics such as accuracy, precision, recall, and F1-score.
    - Compare the results of different models to identify the best-performing one.

5. **Deployment**
    - Develop a Streamlit web application to allow users to test the model with Hindi speech samples.

---

## Directory Structure

```
ROOT
├── artifacts/dataset
├── config
├── emotion
├── models/saved_models
├── notebooks
│   ├── comp_trials.ipynb
│   ├── speech_emotion_recognition.ipynb
├── pipelines
│   ├── stage_01_ingestion.py
│   ├── stage_02_preprocessor.py
│   ├── stage_03_base_models.py
│   ├── stage_04_model_trainer.py
│   ├── stage_05_model_evaluator.py
│   ├── stage_06_predictor.py
├── scripts
│   ├── script_reorg_data.py
│   ├── verify_struct.py
├── .gitattributes
├── .gitignore
├── app.py
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── RESEARCH.md
├── template.py
```

---

## Implementation Details

### Data Preparation
- **Dataset**: Hindi speech dataset containing labeled emotions.
- **Feature Extraction**: Used MFCCs to extract features from audio data.
- **Data Augmentation**: Applied noise addition, pitch shifting, and time stretching to increase dataset diversity.

### Models
The following models were implemented and evaluated:

1. **Convolutional Neural Network (CNN)**
   - Best suited for identifying patterns in MFCCs.
   - Accuracy: 56%

2. **Multi-Layer Perceptron (MLP)**
   - A fully connected network to classify the MFCC features.
   - Accuracy: 56%

3. **Random Forest Classifier**
   - Ensemble-based classifier.
   - Accuracy: 64%

4. **K-Nearest Neighbors (KNN)**
   - Non-parametric model.
   - Accuracy: 47%

### Evaluation Metrics
Models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Challenges
1. **Limited Availability of Hindi Speech Data**
   - Created a synthetic dataset to augment real data.
2. **Noise in Speech Samples**
   - Preprocessed audio to remove background noise.
3. **Feature Extraction**
   - Balancing computation cost and information retention while extracting MFCCs.

---

## Streamlit Application
The Streamlit app allows users to:
- Upload Hindi speech samples.
- View the predicted emotion.
- Access visualizations of MFCC features.

Run the app locally:
```bash
streamlit run app.py
```

---

## Installation and Usage

### Prerequisites
- Python 3.8 or above
- Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Create a virtual environment:
```bash
python -m venv env
```
2. Activate the virtual environment named 'env':
```bash
cd .\env\Scripts\activate
```
3. Run the main script (the prediction pipeline):
```bash
python main.py
```

### Models
Pre-trained models are saved in the `models` directory. You can load them directly for predictions.

---

## Results
- **Best Model**: Random Forest achieved the highest accuracy of 64%.
- **Feature Insights**: MFCC features provided a good representation of speech characteristics.

---

## Future Work
1. Improve model accuracy using advanced architectures like RNN or transformers.
2. Collect and utilize a larger dataset for Hindi speech.
3. Deploy the Streamlit app online for public use.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments
Special thanks to the open-source libraries and contributors whose tools were used in this project.

---

## References
1. Research papers on speech emotion recognition.
2. Dataset (details in `RESEARCH.md`).
