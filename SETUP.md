# ProfileScore Setup Guide

This guide will walk you through setting up the ProfileScore application step by step.

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB+ recommended for better performance)
- Hugging Face account with access to LLaMA 2 models
- Git (for cloning the repository)

## Step 1: Clone and Setup Project

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd ProfileScore

# Or if you have the files locally, navigate to the project directory
cd ProfileScore
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Step 4: Hugging Face Setup

### 4.1 Get Your Access Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "ProfileScore")
4. Select "Read" role
5. Copy the generated token

### 4.2 Request LLaMA 2 Access

1. Go to [Meta's LLaMA 2 page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Click "Request access"
3. Fill out the form and submit
4. Wait for approval (usually takes a few hours to a day)

### 4.3 Login to Hugging Face

```bash
# Login with your token
huggingface-cli login
# Enter your token when prompted
```

## Step 5: Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp env.example .env

# Edit the .env file with your settings
```

Edit the `.env` file with your configuration:

```env
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_actual_token_here

# Model Configuration
MODEL_ID=meta-llama/Llama-2-7b-chat-hf

# Application Configuration
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Optional: Model Parameters
MAX_NEW_TOKENS=200
TEMPERATURE=0.7
TOP_P=0.9
```

## Step 6: Test the Setup

### Option A: Test Direct Service Usage

```bash
# Run the example script
python example_usage.py
```

### Option B: Test API Server

```bash
# Start the API server
python start.py

# In another terminal, run the test script
python test_api.py
```

## Step 7: Verify Installation

### Check Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Test Profile Scoring

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_profile": "Senior Software Engineer with 7 years of experience in Python, Django, and AWS.",
    "job_description": "We are looking for a Backend Engineer with 5+ years of experience in Python, Django/Flask."
  }'
```

## Troubleshooting

### Common Issues

#### 1. "HUGGINGFACE_TOKEN environment variable is required"

**Solution**: Make sure you've created the `.env` file and set your token correctly.

#### 2. "Failed to load LLaMA 2 model"

**Possible causes**:
- Insufficient RAM (need at least 8GB)
- No access to LLaMA 2 models
- Invalid token

**Solutions**:
- Close other applications to free up memory
- Request access to LLaMA 2 models
- Verify your token is correct

#### 3. "CUDA out of memory"

**Solution**: The model is too large for your GPU. Try:
- Using CPU only (slower but works)
- Reducing model size (use 7B instead of 13B)
- Closing other GPU applications

#### 4. Slow model loading

**Solutions**:
- Use an SSD for faster disk I/O
- Ensure stable internet connection for model download
- Consider using a smaller model variant

### Performance Optimization

#### For Better Performance:

1. **Use GPU** (if available):
   - Install CUDA-compatible PyTorch
   - Ensure sufficient GPU memory

2. **Adjust Model Parameters**:
   ```env
   MAX_NEW_TOKENS=150  # Reduce for faster responses
   TEMPERATURE=0.5     # Lower for more consistent results
   ```

3. **Memory Optimization**:
   - Close unnecessary applications
   - Use model quantization if needed

## Next Steps

Once setup is complete:

1. **Explore the API**: Visit `http://localhost:8000/docs` for interactive API documentation
2. **Customize Prompts**: Modify the prompt template in `app/services/llama_service.py`
3. **Add Features**: Extend the application with additional endpoints or functionality
4. **Deploy**: Consider deploying to a cloud platform for production use

## Support

If you encounter issues:

1. Check the logs in the `logs/` directory
2. Verify your Hugging Face token and model access
3. Ensure sufficient system resources
4. Review the troubleshooting section above

For additional help, check the project documentation or create an issue in the repository. 