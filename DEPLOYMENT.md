# 🚀 Deployment Guide

## Overview

This Battery RUL Prediction application is built with Streamlit and can be deployed to various platforms. While Vercel was mentioned, Streamlit apps are best deployed on platforms that support Python applications.

## Recommended Deployment Platforms

### 1. Streamlit Community Cloud ⭐ (Recommended)

**Pros:**
- Free hosting for Streamlit apps
- Zero configuration needed
- Automatic updates from GitHub
- Built-in resource management
- Easy sharing with custom URLs

**Steps:**
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Click "Deploy"

**Configuration:**
- Main file: `app.py`
- Python version: 3.10+ (see `runtime.txt`)
- Requirements: Auto-detected from `requirements.txt`

### 2. Heroku

**Pros:**
- Supports Python apps
- Free tier available
- Easy scaling
- Add-ons for databases

**Steps:**
1. Install Heroku CLI
```bash
heroku login
```

2. Create Heroku app
```bash
heroku create battery-rul-app
```

3. Deploy
```bash
git push heroku main
```

**Required Files (Already Included):**
- `Procfile`: Defines web process
- `setup.sh`: Streamlit configuration
- `runtime.txt`: Python version

### 3. Docker Deployment

**Pros:**
- Runs anywhere Docker is supported
- Consistent environment
- Easy local testing

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run:**
```bash
docker build -t battery-rul-app .
docker run -p 8501:8501 battery-rul-app
```

### 4. AWS EC2

**Pros:**
- Full control
- Scalable
- Production-ready

**Steps:**
1. Launch EC2 instance (Ubuntu 20.04+)
2. SSH into instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip supervisor
```

4. Clone repository
```bash
git clone <your-repo>
cd battery-rul-app
```

5. Install Python packages
```bash
pip3 install -r requirements.txt
```

6. Configure supervisor (already included)
```bash
sudo cp /path/to/streamlit.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start streamlit
```

7. Configure reverse proxy (Nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 5. Google Cloud Platform (Cloud Run)

**Pros:**
- Serverless
- Auto-scaling
- Pay per use

**Steps:**
1. Create `Dockerfile` (see Docker section)
2. Build and push to Container Registry
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/battery-rul-app
```

3. Deploy to Cloud Run
```bash
gcloud run deploy battery-rul-app \
  --image gcr.io/PROJECT-ID/battery-rul-app \
  --platform managed \
  --port 8501
```

## Important Notes About Vercel

⚠️ **Vercel is NOT recommended** for Streamlit applications because:

1. **Architecture Mismatch**: Vercel is optimized for Node.js/Next.js serverless functions
2. **Python Support Limited**: Vercel's Python support is for serverless functions, not long-running apps
3. **WebSocket Requirements**: Streamlit uses WebSockets which don't work well on Vercel
4. **Execution Time Limits**: Vercel has strict timeout limits (10s for free tier)

**If you must use Vercel**, consider:
- Converting to a Next.js frontend with Python API backend
- Using Vercel for static landing page + separate Python backend on another platform

## Environment Variables

For production deployment, configure these environment variables:

```bash
# Optional: NASA API configuration
NASA_API_KEY=your_api_key_here

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

## Resource Requirements

**Minimum:**
- RAM: 2GB
- CPU: 1 core
- Disk: 2GB

**Recommended:**
- RAM: 4GB (for LSTM training)
- CPU: 2 cores
- Disk: 5GB

## Performance Optimization

1. **Model Caching**: Pre-train models and commit to repository
```python
# Run once locally
python -c "from utils.data_processor import load_nasa_dataset; load_nasa_dataset()"
```

2. **Data Caching**: Include sample dataset in repository
```bash
# Sample data already generated in /app/data/
```

3. **Streamlit Caching**: Already configured with `@st.cache_resource` and `@st.cache_data`

4. **Reduce Model Complexity** for resource-constrained environments:
```python
# In ml_models.py, reduce:
- n_estimators (100 -> 50)
- LSTM epochs (50 -> 20)
```

## Security Considerations

1. **Authentication**: Current demo uses hardcoded credentials
   - For production, integrate with OAuth (Google, GitHub)
   - Or use Streamlit's built-in authentication (Enterprise)

2. **HTTPS**: Enable SSL/TLS in production
   - Use Let's Encrypt for free certificates
   - Configure reverse proxy (Nginx/Apache)

3. **Environment Variables**: Never commit API keys
```bash
# Use .env file (already in .gitignore)
echo "NASA_API_KEY=xxx" > .env
```

4. **Input Validation**: Already implemented
   - Parameter ranges enforced with sliders
   - CSV validation on upload

## Monitoring

1. **Logs**: 
```bash
# Supervisor logs
tail -f /var/log/supervisor/streamlit.*.log

# Streamlit logs
~/.streamlit/logs/
```

2. **Health Check**: Run periodically
```bash
python /app/healthcheck.py
```

3. **Metrics**: Consider adding:
   - Application Performance Monitoring (APM)
   - Error tracking (Sentry)
   - Usage analytics (Google Analytics)

## Scaling

### Vertical Scaling
- Increase instance size for more RAM/CPU
- Useful for LSTM training

### Horizontal Scaling
- Run multiple Streamlit instances
- Use load balancer (Nginx, HAProxy)
- Share session state via Redis

### Database
- Current: In-memory (session state)
- Production: Consider PostgreSQL for persistent storage

## Troubleshooting

### App Won't Start
```bash
# Check logs
tail -50 /var/log/supervisor/streamlit.err.log

# Verify dependencies
pip install -r requirements.txt

# Test manually
streamlit run app.py
```

### Out of Memory
- Reduce batch sizes
- Disable LSTM model
- Increase instance RAM
- Use model checkpointing

### Slow Performance
- Pre-train and cache models
- Use sample dataset instead of full NASA dataset
- Enable Streamlit caching
- Use CDN for static assets

## Backup and Recovery

1. **Code**: Version control with Git
2. **Models**: Save to cloud storage (S3, GCS)
3. **Data**: Backup `/app/data/` directory
4. **Configuration**: Document environment variables

## CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Streamlit Cloud
        run: |
          # Streamlit Cloud auto-deploys on push
          echo "Deployed!"
```

## Cost Estimation

**Streamlit Cloud**: Free (Community tier)

**Heroku**: 
- Free tier: $0
- Hobby: $7/month
- Production: $25+/month

**AWS EC2**:
- t3.small: ~$15/month
- t3.medium: ~$30/month

**GCP Cloud Run**:
- Pay per use: ~$5-20/month for moderate traffic

## Support

For deployment issues:
1. Check logs
2. Review this guide
3. Consult platform documentation
4. Open GitHub issue

---

**Ready to deploy? Choose your platform and follow the steps above!** 🚀
