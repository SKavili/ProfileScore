# ProfileScore - Advanced LLaMA 2 Job Profile Matching

A comprehensive Python backend service that uses Meta's LLaMA 2 model with Pinecone vector storage and LangChain integration for advanced job profile matching and scoring.

## 🚀 Features

- 🤖 **LLaMA 2-powered scoring** with intelligent prompt engineering
- 🗄️ **Pinecone vector storage** for efficient candidate and job data management
- 🔗 **LangChain integration** for enhanced AI processing and analysis
- 📊 **Enhanced scoring** with skill matching, experience analysis, and confidence metrics
- 🔍 **Vector similarity search** for finding the best candidates
- 📝 **Comprehensive logging** and monitoring
- 🧪 **Batch processing** for efficient large-scale scoring
- 🔄 **Scoring history** tracking and retrieval
- 🏗️ **Modular architecture** with clean separation of concerns

## 🏗️ Architecture

```
ProfileScore/
├── app/
│   ├── main.py              # FastAPI application with all endpoints
│   ├── models.py            # Pydantic models for data validation
│   ├── services/
│   │   ├── llama_service.py     # LLaMA 2 model integration
│   │   ├── pinecone_service.py  # Pinecone vector storage
│   │   └── langchain_service.py # LangChain enhanced processing
│   └── utils/
│       └── logger.py        # Comprehensive logging
├── requirements.txt         # All dependencies
├── start.py                # Application startup script
├── test_enhanced_api.py    # Comprehensive test suite
└── README.md
```

## 📋 Prerequisites

- Python 3.8+
- Hugging Face account with LLaMA 2 access
- Pinecone account and API key
- Sufficient RAM (8GB+ recommended for 7B model)

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file based on `env.example`:

```env
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model Configuration
MODEL_ID=meta-llama/Llama-2-7b-chat-hf

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=profilescores

# LangChain Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Vector Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Setup Services

1. **Hugging Face**: Get your token and request LLaMA 2 access
2. **Pinecone**: Create an account and get your API key
3. **LangChain**: Optional - for enhanced tracing and monitoring

### 4. Run the Application

```bash
python start.py
```

## 📚 API Endpoints

### Core Scoring

#### Basic Scoring
```bash
POST /score
{
  "candidate_profile": "Senior Software Engineer with 7 years...",
  "job_description": "We are looking for a Backend Engineer...",
  "candidate_id": "candidate_123",
  "job_id": "job_456"
}
```

#### Enhanced Scoring (LangChain)
```bash
POST /score/enhanced
{
  "candidate_profile": "Senior Software Engineer with 7 years...",
  "job_description": "We are looking for a Backend Engineer...",
  "candidate_id": "candidate_123",
  "job_id": "job_456"
}
```

### Candidate Management

#### Create Candidate
```bash
POST /candidates
{
  "candidate_id": "candidate_123",
  "name": "John Doe",
  "email": "john@example.com",
  "profile_text": "Senior Software Engineer...",
  "skills": ["Python", "Django", "AWS"],
  "experience_years": 7,
  "current_role": "Senior Software Engineer",
  "location": "San Francisco, CA"
}
```

#### Search Candidates
```bash
POST /candidates/search
{
  "query": "Python backend engineer",
  "max_results": 10,
  "min_score": 70,
  "skills_filter": ["Python", "Django"],
  "location_filter": "San Francisco"
}
```

#### Get Candidate
```bash
GET /candidates/{candidate_id}
```

### Job Management

#### Create Job
```bash
POST /jobs
{
  "job_id": "job_456",
  "title": "Senior Backend Engineer",
  "company": "TechCorp",
  "description": "We are looking for...",
  "required_skills": ["Python", "Django"],
  "preferred_skills": ["Machine Learning"],
  "experience_required": 5,
  "location": "San Francisco, CA"
}
```

#### Get Job
```bash
GET /jobs/{job_id}
```

### Batch Processing

#### Batch Scoring
```bash
POST /batch/score
{
  "candidates": [
    "Junior developer with 2 years of Python experience",
    "Senior engineer with 8 years of full-stack development"
  ],
  "job_description": "We are looking for..."
}
```

### Analytics & History

#### Scoring History
```bash
GET /scoring/history?candidate_id=candidate_123&job_id=job_456&limit=50
```

#### System Information
```bash
GET /health
GET /model/info
```

## 🔍 Enhanced Features

### LangChain Integration

- **Skill Extraction**: Automatically extract technical skills from profiles
- **Profile Analysis**: Comprehensive candidate analysis including experience, education, achievements
- **Job Requirements Extraction**: Parse job descriptions for required/preferred skills
- **Enhanced Scoring**: Multi-factor scoring combining AI analysis, skill matching, and experience alignment
- **Batch Processing**: Efficient processing of multiple candidates

### Pinecone Vector Storage

- **Vector Embeddings**: Store candidate and job data as high-dimensional vectors
- **Similarity Search**: Find similar candidates using vector similarity
- **Metadata Filtering**: Filter results by skills, location, experience, etc.
- **Scoring History**: Track and retrieve historical scoring results
- **Scalable Storage**: Handle large volumes of candidate and job data

### Advanced Scoring Algorithm

The enhanced scoring combines multiple factors:

1. **Base AI Score (60%)**: LLaMA 2's comprehensive analysis
2. **Skill Match Score (30%)**: Technical skills alignment
3. **Experience Match Score (10%)**: Years of experience comparison

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_enhanced_api.py
```

This will test:
- Basic and enhanced scoring
- Candidate and job creation
- Vector similarity search
- Batch processing
- Scoring history retrieval

## 📊 Response Examples

### Enhanced Scoring Response
```json
{
  "enhanced_score": 92,
  "base_score": 88,
  "skill_match_score": 95,
  "experience_match_score": 100,
  "explanation": "The candidate demonstrates excellent alignment...",
  "confidence": 0.85,
  "candidate_skills": ["Python", "Django", "AWS", "PostgreSQL"],
  "required_skills": ["Python", "Django", "AWS"],
  "preferred_skills": ["Machine Learning"],
  "skill_matches": {
    "required": 3,
    "preferred": 1,
    "total_required": 3,
    "total_preferred": 1
  },
  "candidate_analysis": {
    "experience_years": 7,
    "current_role": "Senior Software Engineer",
    "location": "San Francisco, CA"
  },
  "job_requirements": {
    "required_skills": ["Python", "Django", "AWS"],
    "experience_required": 5
  }
}
```

### Candidate Search Response
```json
{
  "candidates": [
    {
      "candidate_id": "candidate_123",
      "name": "John Doe",
      "email": "john@example.com",
      "skills": ["Python", "Django", "AWS"],
      "experience_years": 7,
      "current_role": "Senior Software Engineer",
      "location": "San Francisco, CA",
      "similarity_score": 0.92
    }
  ],
  "total_results": 1,
  "query": "Python backend engineer",
  "search_metadata": {
    "filters_applied": {
      "min_score": 70,
      "skills_filter": ["Python", "Django"]
    }
  }
}
```

## 🔧 Configuration

### Model Parameters
- `MAX_NEW_TOKENS`: Maximum tokens for model responses (default: 200)
- `TEMPERATURE`: Response creativity (default: 0.7)
- `TOP_P`: Nucleus sampling parameter (default: 0.9)

### Vector Storage
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `PINECONE_INDEX_NAME`: Pinecone index name
- Vector dimension: 384 (for all-MiniLM-L6-v2)

## 🚀 Performance Optimization

### For Production Use

1. **GPU Acceleration**: Use CUDA-compatible PyTorch for faster inference
2. **Model Quantization**: Consider quantized models for memory efficiency
3. **Batch Processing**: Use batch endpoints for large-scale operations
4. **Caching**: Implement Redis caching for frequently accessed data
5. **Load Balancing**: Deploy multiple instances behind a load balancer

### Monitoring

- **Health Checks**: Monitor `/health` endpoint
- **Model Performance**: Track scoring accuracy and response times
- **Vector Storage**: Monitor Pinecone index performance
- **Error Tracking**: Comprehensive logging with loguru

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project uses LLaMA 2 which is subject to Meta's license terms. Please review the [LLaMA 2 license](https://github.com/facebookresearch/llama/blob/main/LICENSE) for usage restrictions.

## 🆘 Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Verify your API keys and service connections
3. Review the troubleshooting section in `SETUP.md`
4. Create an issue in the repository

---

**ProfileScore** - Advanced AI-powered job profile matching with vector storage and enhanced processing capabilities. 