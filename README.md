# AI Healthcare Diagnostics

## Problem Statement
Early disease diagnosis is critical in healthcare. This project aims to build an AI-driven diagnostic system using machine learning and explainable AI.

## Tech Stack
- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn, Random Forest
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Generative AI**: OpenAI/Groq APIs (planned)
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Deployment**: Render

## Features
- Disease prediction for Diabetes, Heart Disease, Pneumonia, Kidney Disease, Liver Disease
- Explainable AI with feature impact analysis
- Interactive web interface
- Symptom checker
- Model training scripts

## Project Status
✅ Core functionality implemented
✅ SHAP explanations working
✅ Web interface deployed
🚧 Generative AI integration pending

## Local Development

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-healthcare-diagnostics

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/app.py
```

Visit `http://localhost:5000` in your browser.

## Deployment on Render

### Automatic Deployment (Recommended)
1. Fork/clone this repository to GitHub
2. Create a Render account at https://render.com
3. Connect your GitHub repository to Render
4. Render will automatically detect the `render.yaml` configuration
5. Set environment variables in Render dashboard:
   - `FLASK_APP=src/app.py`
   - `FLASK_ENV=production`
   - `FLASK_RUN_HOST=0.0.0.0`
   - `FLASK_RUN_PORT=10000`
6. Deploy!

The application uses **Gunicorn** as the production WSGI server for better performance and stability.

### Manual Deployment
If you prefer to use the Dockerfile:
1. Follow steps 1-3 above
2. Choose "Docker" as the runtime
3. Render will use the provided Dockerfile (also configured with Gunicorn)

### Environment Variables
Set these in your Render service settings:
- `FLASK_APP`: `src/app.py`
- `FLASK_ENV`: `production`
- `FLASK_RUN_HOST`: `0.0.0.0`
- `FLASK_RUN_PORT`: `10000`

## API Endpoints
- `GET /`: Home page
- `POST /predict/diabetes`: Diabetes prediction
- `POST /predict/heart`: Heart disease prediction
- `POST /predict/pneumonia`: Pneumonia prediction
- `POST /predict/kidney`: Kidney disease prediction
- `POST /predict/liver`: Liver disease prediction
- `POST /symptom-checker`: Symptom analysis

## Model Training
Run the training script to retrain models:
```bash
python train_models.py
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License
See LICENSE file for details.
