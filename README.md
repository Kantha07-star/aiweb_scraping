FastAPI AI Website Business Intelligence API

Overview:  
This FastAPI application scrapes and analyzes website homepage content to extract business-related information using Google's Gemini AI or a rule-based fallback system.   
It accepts a website URL and optional user-defined questions and returns structured company information and answers derived from the content.

Features:
Token-based Authentication  
Website scraping using BeautifulSoup  
AI-powered analysis with Google Gemini (if enabled)  
Rule-based analysis fallback    
Customizable questions  
Extracts:  
  Industry  
  Company Size  
  Location  
  Description  
  Confidence Score  
Robust error handling  
Health Check Endpoints  
  
Installation  
pip install fastapi uvicorn google-generativeai beautifulsoup4 requests pydantic python-multipart python-dotenv  
  
Ensure you have a .env file with the following variables:  
API_SECRET_KEY=your_api_secret_key GOOGLE_API_KEY=your_google_api_key ENABLE_AI=true  
  
Running the App  
uvicorn app:app --reload  
  
Authentication  
All endpoints (except health checks) require a Bearer token passed via the Authorization header:  
  
Authorization: Bearer your_api_secret_key  
  
API Endpoints  
GET /  
Root Health Check  
Returns the server status and configuration summary.  
  
GET /health  
Detailed Health Check  
Returns:  
Server status  
AI availability  
Security configuration  
    
POST /analyze  
Analyze a Website  
Request Body:  
{  
"url": "https://example.com",  
"questions": [  
"What industry does this company belong to?",  
"Where is it located?"  
]  
}  
  
Response:  
{  
"url": "https://example.com",  
"title": "Example Domain",  
"company_info": {  
"industry": "Technology",  
"company_size": "Medium",  
"location": "San Francisco, CA",  
"description": "A leading provider of cloud solutions.",  
"confidence_score": 0.88  
},  
"questions_answered": {  
"What industry does this company belong to?": "Technology",  
"Where is it located?": "San Francisco, CA"  
},  
"processing_time_seconds": 4.52, 
"success": true,  
"ai_model_used": "google-gemini"  
}  
  
AI & Rule-Based Logic  
Gemini AI is used if ENABLE_AI=true and GOOGLE_API_KEY is valid.  
If Gemini fails or is not configured, a rule-based extractor is used.  
Location extraction includes schema.org parsing, heuristics, and known location matching.  
Text processing is optimized for performance and clarity.  
  
Folder Structure  
project/  
├── app.py # Main FastAPI application  
├── .env # Environment variables  
└── requirements.txt # Dependency list (optional)  
  
Testing the API  
Use tools like Postman, curl, or the built-in Swagger UI at: http://localhost:8000/docs  
  
Example .env  
API_SECRET_KEY=supersecure123  
GOOGLE_API_KEY=your_google_key_here  
ENABLE_AI=true  
  
Error Handling  
Standardized ErrorResponse includes:  
error_code  
message  
timestamp  
Example: {  
"success": false,  
"error": {  
"error_code": "401",  
"message": "Invalid authentication credentials."  
},  
"timestamp": "2025-06-13T20:01:10.222Z"  
}  
