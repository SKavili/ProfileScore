# ProfileScore End-to-End Class Diagram

## UML Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           ProfileScore Class Diagram                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           ENUMERATIONS                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CandidateStatus                                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ACTIVE: str                                                                                               │
│  INACTIVE: str                                                                                              │
│  HIRED: str                                                                                                 │
│  REJECTED: str                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           DATA MODELS (Pydantic)                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ScoringRequest                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + candidate_profile: str                                                                                   │
│  + job_description: str                                                                                     │
│  + candidate_id: Optional[str]                                                                              │
│  + job_id: Optional[str]                                                                                    │
│  + metadata: Optional[Dict[str, Any]]                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + validate_text_content(cls, v) -> str                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ uses
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ScoringResponse                                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + score: int                                                                                               │
│  + explanation: str                                                                                         │
│  + confidence: float                                                                                        │
│  + timestamp: datetime                                                                                      │
│  + model_used: str                                                                                          │
│  + scoring_id: Optional[str]                                                                                │
│  + candidate_id: Optional[str]                                                                              │
│  + job_id: Optional[str]                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CandidateData                                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + candidate_id: str                                                                                        │
│  + name: Optional[str]                                                                                      │
│  + email: Optional[str]                                                                                     │
│  + profile_text: str                                                                                        │
│  + skills: List[str]                                                                                        │
│  + experience_years: Optional[int]                                                                          │
│  + current_role: Optional[str]                                                                              │
│  + location: Optional[str]                                                                                  │
│  + status: CandidateStatus                                                                                  │
│  + created_at: datetime                                                                                     │
│  + updated_at: datetime                                                                                     │
│  + metadata: Dict[str, Any]                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ uses
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  JobData                                                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + job_id: str                                                                                              │
│  + title: str                                                                                               │
│  + company: str                                                                                             │
│  + description: str                                                                                         │
│  + required_skills: List[str]                                                                               │
│  + preferred_skills: List[str]                                                                              │
│  + experience_required: Optional[int]                                                                       │
│  + location: Optional[str]                                                                                  │
│  + salary_range: Optional[str]                                                                              │
│  + is_active: bool                                                                                          │
│  + created_at: datetime                                                                                     │
│  + updated_at: datetime                                                                                     │
│  + metadata: Dict[str, Any]                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ScoringResult                                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + scoring_id: str                                                                                          │
│  + candidate_id: str                                                                                        │
│  + job_id: str                                                                                              │
│  + score: int                                                                                               │
│  + explanation: str                                                                                         │
│  + confidence: float                                                                                        │
│  + model_used: str                                                                                          │
│  + created_at: datetime                                                                                     │
│  + metadata: Dict[str, Any]                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CandidateSearchRequest                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + query: str                                                                                               │
│  + job_id: Optional[str]                                                                                    │
│  + min_score: Optional[int]                                                                                 │
│  + max_results: int                                                                                         │
│  + skills_filter: Optional[List[str]]                                                                       │
│  + location_filter: Optional[str]                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ uses
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CandidateSearchResponse                                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + candidates: List[Dict[str, Any]]                                                                         │
│  + total_results: int                                                                                       │
│  + query: str                                                                                               │
│  + search_metadata: Dict[str, Any]                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  HealthResponse                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + status: str                                                                                              │
│  + timestamp: datetime                                                                                      │
│  + version: str                                                                                             │
│  + model_loaded: bool                                                                                       │
│  + pinecone_connected: bool                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ErrorResponse                                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + error: str                                                                                               │
│  + detail: Optional[str]                                                                                    │
│  + timestamp: datetime                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           SERVICE LAYER                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LLaMAService                                                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  - logger: Logger                                                                                           │
│  - model: Optional[AutoModelForCausalLM]                                                                   │
│  - tokenizer: Optional[AutoTokenizer]                                                                       │
│  - pipeline: Optional[Pipeline]                                                                             │
│  - model_id: str                                                                                            │
│  - is_loaded: bool                                                                                          │
│  - max_new_tokens: int                                                                                      │
│  - temperature: float                                                                                       │
│  - top_p: float                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + __init__()                                                                                               │
│  + load_model() -> bool                                                                                     │
│  + score_profile(candidate_profile: str, job_description: str) -> ScoringResponse                          │
│  + get_model_info() -> Dict[str, Any]                                                                       │
│  + unload_model() -> None                                                                                   │
│  - _create_scoring_prompt(candidate_profile: str, job_description: str) -> str                             │
│  - _extract_score_from_response(response: str) -> Tuple[int, str, float]                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ uses
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PineconeService                                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  - logger: Logger                                                                                           │
│  - index: Optional[Index]                                                                                   │
│  - embedding_model: Optional[SentenceTransformer]                                                           │
│  - is_connected: bool                                                                                       │
│  - api_key: str                                                                                             │
│  - environment: str                                                                                         │
│  - index_name: str                                                                                          │
│  - embedding_model_name: str                                                                                │
│  - vector_dimension: int                                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + __init__()                                                                                               │
│  + connect() -> bool                                                                                        │
│  + store_candidate(candidate_data: CandidateData) -> bool                                                   │
│  + store_job(job_data: JobData) -> bool                                                                     │
│  + store_scoring_result(scoring_result: ScoringResult) -> bool                                              │
│  + search_candidates(query: str, ...) -> List[Dict[str, Any]]                                               │
│  + get_candidate(candidate_id: str) -> Optional[Dict[str, Any]]                                             │
│  + get_job(job_id: str) -> Optional[Dict[str, Any]]                                                         │
│  + get_scoring_history(...) -> List[Dict[str, Any]]                                                         │
│  + delete_candidate(candidate_id: str) -> bool                                                              │
│  + delete_job(job_id: str) -> bool                                                                          │
│  + get_index_stats() -> Dict[str, Any]                                                                      │
│  - _generate_embedding(text: str) -> List[float]                                                            │
│  - _generate_id(prefix: str) -> str                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ uses
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LangChainService                                                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  - logger: Logger                                                                                           │
│  - llm: Optional[HuggingFacePipeline]                                                                       │
│  - embeddings: Optional[HuggingFaceEmbeddings]                                                              │
│  - text_splitter: RecursiveCharacterTextSplitter                                                            │
│  - vectorstore: Optional[Any]                                                                               │
│  - is_initialized: bool                                                                                     │
│  - embedding_model_name: str                                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + __init__()                                                                                               │
│  + initialize() -> bool                                                                                     │
│  + extract_skills(profile_text: str) -> List[str]                                                           │
│  + analyze_profile(profile_text: str) -> Dict[str, Any]                                                     │
│  + extract_job_requirements(job_description: str) -> Dict[str, Any]                                         │
│  + enhanced_scoring(candidate_profile: str, job_description: str) -> Dict[str, Any]                        │
│  + create_search_tools() -> List[Tool]                                                                      │
│  + create_agent() -> Any                                                                                    │
│  + process_batch(candidates: List[str], job_description: str) -> List[Dict[str, Any]]                      │
│  - _fallback_skill_extraction(profile_text: str) -> List[str]                                              │
│  - _fallback_profile_analysis(profile_text: str) -> Dict[str, Any]                                         │
│  - _fallback_job_requirements(job_description: str) -> Dict[str, Any]                                       │
│  - _search_by_skills(skills_query: str) -> str                                                              │
│  - _search_by_experience(experience_query: str) -> str                                                      │
│  - _search_by_location(location_query: str) -> str                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           UTILITY LAYER                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ConfigValidator                                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  - logger: Logger                                                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + __init__()                                                                                               │
│  + validate_required_env_vars() -> Dict[str, bool]                                                          │
│  + validate_optional_env_vars() -> Dict[str, bool]                                                          │
│  + get_config_summary() -> Dict[str, Any]                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           FASTAPI APPLICATION                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI App (main.py)                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  - app: FastAPI                                                                                             │
│  - logger: Logger                                                                                           │
│  - llama_service: LLaMAService                                                                              │
│  - pinecone_service: PineconeService                                                                        │
│  - langchain_service: LangChainService                                                                      │
│  - config_validator: ConfigValidator                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  + lifespan(app: FastAPI) -> AsyncContextManager                                                            │
│  + root() -> dict                                                                                           │
│  + health_check() -> HealthResponse                                                                         │
│  + score_profile(request: ScoringRequest) -> ScoringResponse                                                │
│  + enhanced_score_profile(request: ScoringRequest) -> dict                                                  │
│  + create_candidate(candidate_data: CandidateData) -> dict                                                  │
│  + get_candidate(candidate_id: str) -> dict                                                                 │
│  + search_candidates(request: CandidateSearchRequest) -> CandidateSearchResponse                            │
│  + create_job(job_data: JobData) -> dict                                                                    │
│  + get_job(job_id: str) -> dict                                                                             │
│  + get_scoring_history(...) -> dict                                                                         │
│  + batch_score_profiles(request: dict) -> dict                                                              │
│  + get_model_info() -> dict                                                                                 │
│  + get_configuration() -> dict                                                                              │
│  + reload_model() -> dict                                                                                   │
│  + global_exception_handler(request, exc) -> JSONResponse                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           RELATIONSHIPS                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Relationship Types:                                                                                        │
│                                                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                                │
│  │   Uses      │    │  Creates    │    │  Validates  │    │  Stores     │                                │
│  │   ──────►   │    │   ──────►   │    │   ──────►   │    │   ──────►   │                                │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘                                │
│                                                                                                             │
│  Key Relationships:                                                                                         │
│                                                                                                             │
│  • FastAPI App ──────► LLaMAService (uses for AI scoring)                                                  │
│  • FastAPI App ──────► PineconeService (uses for vector storage)                                          │
│  • FastAPI App ──────► LangChainService (uses for enhanced processing)                                    │
│  • FastAPI App ──────► ConfigValidator (uses for configuration validation)                                 │
│                                                                                                             │
│  • LLaMAService ──────► ScoringRequest (creates from)                                                      │
│  • LLaMAService ──────► ScoringResponse (creates)                                                          │
│                                                                                                             │
│  • PineconeService ──────► CandidateData (stores)                                                          │
│  • PineconeService ──────► JobData (stores)                                                                │
│  • PineconeService ──────► ScoringResult (stores)                                                          │
│                                                                                                             │
│  • LangChainService ──────► LLaMAService (uses for LLM)                                                    │
│  • LangChainService ──────► PineconeService (uses for vector search)                                       │
│                                                                                                             │
│  • ConfigValidator ──────► Environment Variables (validates)                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           EXTERNAL DEPENDENCIES                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  External Libraries:                                                                                        │
│                                                                                                             │
│  • FastAPI ──────► Web framework for API endpoints                                                         │
│  • Pydantic ──────► Data validation and serialization                                                      │
│  • Transformers ──────► Hugging Face transformers library                                                   │
│  • SentenceTransformers ──────► Text embedding generation                                                   │
│  • Pinecone ──────► Vector database client                                                                 │
│  • LangChain ──────► AI processing framework                                                                │
│  • Loguru ──────► Structured logging                                                                       │
│  • PyTorch ──────► Deep learning framework                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

## Key Design Patterns

### **1. Service Layer Pattern**
- **Separation of Concerns**: Each service handles a specific domain
- **Loose Coupling**: Services are independent and can be tested separately
- **Single Responsibility**: Each service has one clear purpose

### **2. Repository Pattern**
- **PineconeService**: Acts as a repository for vector data storage
- **Data Access Abstraction**: Hides the complexity of vector database operations
- **CRUD Operations**: Standard create, read, update, delete operations

### **3. Factory Pattern**
- **Model Loading**: LLaMAService creates and manages AI model instances
- **Configuration Management**: ConfigValidator creates configuration objects
- **Service Initialization**: Centralized service creation and management

### **4. Strategy Pattern**
- **Multiple Scoring Strategies**: Basic scoring vs enhanced scoring
- **Fallback Mechanisms**: Multiple approaches for skill extraction
- **Model Selection**: Different models for different tasks

### **5. Observer Pattern**
- **Logging**: All services use centralized logging
- **Health Monitoring**: Application monitors service health
- **Error Handling**: Global exception handling across the application

## Data Flow Summary

1. **Request Flow**: Client → FastAPI → Service Layer → External APIs
2. **Data Flow**: Pydantic Models → Services → Vector Database
3. **AI Flow**: Text Input → LLaMA Service → Scoring Response
4. **Search Flow**: Query → Pinecone Service → Vector Search → Results
5. **Enhancement Flow**: Text → LangChain Service → Enhanced Analysis

This class diagram shows a well-structured, modular system with clear separation of concerns and robust error handling! 🎯 