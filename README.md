# 🔋 Battery RUL Prediction System

AI-Powered Remaining Useful Life (RUL) prediction for lithium-ion batteries in electric vehicles.

## Features

- 🤖 **Multiple ML Models**: XGBoost, Random Forest, Linear Regression, LSTM
- 📊 **Real-time Predictions**: Input battery parameters or upload CSV data
- 🔍 **Model Explainability**: SHAP integration for feature importance analysis
- 🔬 **What-If Analysis**: Simulate different scenarios to optimize battery life
- 🌍 **Sustainability Tracking**: Monitor environmental impact and cost savings
- 📥 **Export Reports**: Generate CSV and PDF reports
- 🎨 **Modern UI**: Dark theme with interactive visualizations

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd battery-rul-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Login

Use demo credentials:
- Username: `admin` / Password: `battery123`
- Username: `demo` / Password: `demo123`

### Predict RUL

1. Navigate to **Predict RUL** page
2. Input battery parameters (temperature, voltage, current, cycle count)
3. Click **Predict** to get RUL forecast
4. View capacity fade curve and sustainability impact
5. Export results as CSV or PDF

### Train Models

1. Navigate to **Train Models** page
2. View current model performance metrics
3. Compare models using interactive charts
4. Retrain models on new data if needed

### What-If Analysis

1. Navigate to **What-If Analysis** page
2. Set base scenario parameters
3. Run analysis to see how different conditions affect RUL
4. Get recommendations for optimizing battery life

## Dataset

The application uses the NASA Randomized Battery Usage Dataset with automatic download fallback. If the download fails, synthetic sample data is generated for demonstration.

## Models

### XGBoost (Primary)
- Best overall performance (MAE < 40 cycles)
- Optimized hyperparameters for battery data
- Fast inference time

### Random Forest
- Robust ensemble method
- Good generalization
- Feature importance insights

### Linear Regression
- Baseline model for comparison
- Fast training and inference
- Interpretable coefficients

### LSTM
- Deep learning approach
- Captures sequential patterns
- Pre-trained and cached

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── utils/
│   ├── auth.py           # Authentication module
│   ├── data_processor.py # Data loading and preprocessing
│   ├── ml_models.py      # Model training and inference
│   ├── explainer.py      # SHAP explainability
│   ├── visualizer.py     # Plotting functions
│   └── report_generator.py # Export functionality
├── models/               # Saved ML models
├── data/                 # Dataset storage
└── reports/              # Generated reports
```

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Heroku

1. Create `setup.sh` and `Procfile` (see deployment docs)
2. Push to Heroku:
```bash
heroku create
git push heroku main
```

### Docker

```bash
docker build -t battery-rul-app .
docker run -p 8501:8501 battery-rul-app
```

## Configuration

Customize theme and settings in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
```

## Technologies

- **Frontend/Backend**: Streamlit
- **ML**: scikit-learn, XGBoost, TensorFlow
- **Data**: pandas, numpy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP
- **Reports**: ReportLab

## Performance

- **Target MAE**: < 40 cycles
- **Inference Time**: < 100ms
- **Training Time**: ~2-5 minutes (first run)
- **Data Processing**: Cached for performance

## Environmental Impact

Accurate RUL prediction helps:
- ♻️ Reduce e-waste by 20%
- 🌍 Lower CO₂ emissions
- 💰 Save replacement costs
- 🔋 Extend battery lifecycle

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Create an issue on GitHub
- Contact: support@example.com

## Acknowledgments

- NASA for the battery dataset
- Streamlit for the amazing framework
- SHAP team for explainability tools
- Open source community

---

**Built with ❤️ for sustainable transportation**