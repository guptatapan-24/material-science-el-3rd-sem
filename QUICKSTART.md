# 🚀 Quick Start Guide

## Getting Started in 5 Minutes

### 1. Access the Application

The Streamlit app is now running at: **http://localhost:8501**

### 2. Login

Use these demo credentials:
- **Username**: `admin`
- **Password**: `battery123`

OR

- **Username**: `demo`
- **Password**: `demo123`

### 3. Navigate the App

After login, you'll see 5 main sections in the sidebar:

#### 🏠 Home
- Overview of features
- Sustainability statistics
- Quick access buttons

#### 📈 Predict RUL
1. Choose input method (Manual or CSV Upload)
2. Enter battery parameters:
   - Temperature: 20-50°C (try 35°C)
   - Voltage: 3.0-4.5V (try 3.7V)
   - Current: -2.0 to 2.0A (try 1.0A)
   - Cycle Count: 0-2000 (try 500)
   - Capacity: 1.0-2.5Ah (try 1.8Ah)
3. Select ML model (XGBoost recommended)
4. Click "Predict RUL"
5. View results, charts, and download reports

#### 🧠 Train Models
- View model performance metrics
- Compare different ML models
- Retrain models on new data

#### 🔬 What-If Analysis
1. Set base scenario parameters
2. Run analysis to see impact of changes
3. Get optimization recommendations

#### ℹ️ About
- Project information
- Technical details
- Credits and resources

### 4. Try a Sample Prediction

**Example Parameters:**
- Temperature: 35°C
- Voltage: 3.7V
- Current: 1.0A
- Cycle Count: 500
- Capacity: 1.8Ah
- Model: XGBoost

Expected RUL: ~400-600 cycles (varies based on trained model)

### 5. Export Results

After prediction:
- Click "Download CSV" for data export
- Click "Download PDF" for formatted report

## Common Tasks

### Running the App Manually

```bash
cd /app
streamlit run app.py
```

### Restarting the App

```bash
sudo supervisorctl restart streamlit
```

### Checking Logs

```bash
tail -f /var/log/supervisor/streamlit.out.log
```

### Viewing Errors

```bash
tail -f /var/log/supervisor/streamlit.err.log
```

## Features at a Glance

| Feature | Description |
|---------|-------------|
| 🤖 **4 ML Models** | XGBoost, Random Forest, Linear Regression, LSTM |
| 📊 **Interactive Charts** | Plotly visualizations for all predictions |
| 🔍 **SHAP Explainability** | Understand feature importance |
| 🔬 **What-If Scenarios** | Test different parameters |
| 🌍 **Sustainability** | Track environmental impact |
| 📥 **Export** | CSV and PDF reports |
| 🎨 **Modern UI** | Dark theme with gradients |
| 🔐 **Authentication** | Session-based login |

## Tips

1. **First Load**: Initial model training may take 2-3 minutes
2. **Model Choice**: XGBoost typically provides best accuracy
3. **Temperature Impact**: Lower temps generally improve battery life
4. **CSV Upload**: Ensure CSV has required columns (see About page)
5. **What-If**: Use to optimize operating conditions

## Troubleshooting

### App Not Loading
```bash
sudo supervisorctl status streamlit
# Should show RUNNING
```

### Import Errors
```bash
pip install -r /app/requirements.txt
```

### Dataset Issues
- App will auto-generate sample data if NASA download fails
- Check `/app/data/` directory for cached data

### Model Training Slow
- Normal for first run (2-5 minutes)
- Models are cached after training
- LSTM training takes longest

## Next Steps

1. ✅ Explore all pages
2. ✅ Try different parameter combinations
3. ✅ Compare model performances
4. ✅ Run what-if scenarios
5. ✅ Generate and download reports
6. ✅ Upload your own battery data (CSV)

## Support

For issues:
1. Check logs: `/var/log/supervisor/streamlit.*.log`
2. Verify all services running: `sudo supervisorctl status`
3. Review README.md for detailed documentation

---

**Enjoy predicting battery health! 🔋⚡**
