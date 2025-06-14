"""
FastAPI AI Agent for Website Scraping with Google Gemini
========================================================

A FastAPI application that uses Google's Gemini AI to extract key business details from website homepages.
Run locally with: uvicorn app:app --reload (or use run.py)

Requirements:
pip install fastapi uvicorn google-generativeai beautifulsoup4 requests pydantic python-multipart python-dotenv
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, Dict, Any, List, Tuple  # Added Tuple
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin, urlparse
import re
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Website Business Intelligence API (Powered by Google Gemini)",
    description="Extract key business details from website homepages using Google's Gemini AI. Provide a website URL and optional questions to get insights.",
    version="2.0.1",  # Incremented version
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("API_SECRET_KEY")
if not SECRET_KEY:
    logger.warning(
        "API_SECRET_KEY not found in environment. Authentication WILL FAIL for protected endpoints. Please set it in your .env file.")
    # Setting a default or allowing startup without it is a security risk for protected endpoints.
    # For this example, we'll proceed but auth will fail if SECRET_KEY remains None and is checked.

# Configuration for Google Gemini
ENABLE_AI = os.getenv("ENABLE_AI", "false").lower() == "true"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if ENABLE_AI:
    try:
        import google.generativeai as genai

        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Google Gemini AI configured with API key.")
        else:
            logger.warning(
                "ENABLE_AI is true, but GOOGLE_API_KEY is not found. AI features will be effectively disabled.")
            ENABLE_AI = False  # Force disable if key is missing
    except ImportError:
        logger.warning("google-generativeai library not installed. AI features will be disabled.")
        ENABLE_AI = False
    except Exception as e:
        logger.error(f"Error configuring Google Gemini: {e}. AI features will be disabled.")
        ENABLE_AI = False

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", 4)))


# Pydantic Models (definitions remain the same as your last provided app.py)
class WebsiteAnalysisRequest(BaseModel):
    """Request model for website analysis"""
    url: HttpUrl = Field(..., examples=["https://www.example.com"],
                         description="The website URL to analyze (must start with http:// or https://)")
    questions: Optional[List[str]] = Field(
        default=[
            "What industry does this company belong to?",
            "What is the approximate size of the company (e.g., small, medium, large, startup)?",
            "Where is the company headquartered or primarily located?"
        ],
        examples=[["What are the main products/services offered?", "Who is the target audience?"]],
        description="List of questions to answer about the website content."
    )

    @field_validator('url')
    @classmethod
    def validate_url_scheme(cls, v: HttpUrl) -> HttpUrl:
        url_str = str(v)
        if not (url_str.startswith('http://') or url_str.startswith('https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class CompanyInfo(BaseModel):
    """Model for company information extracted from website"""
    industry: Optional[str] = Field(None, examples=["Technology"], description="Company industry/sector")
    company_size: Optional[str] = Field(None, examples=["Large"],
                                        description="Approximate company size (e.g., small, medium, large, startup, educational institution size)")
    location: Optional[str] = Field(None, examples=["San Francisco, CA"], description="Company location/headquarters")
    description: Optional[str] = Field(None, examples=["A leading provider of cloud solutions."],
                                       description="Brief company description based on website content")
    confidence_score: Optional[float] = Field(None, examples=[0.85],
                                              description="AI confidence in the extracted company_info (0.0-1.0)")


class WebsiteAnalysisResponse(BaseModel):
    """Response model for website analysis"""
    url: str = Field(..., examples=["https://www.example.com"], description="The analyzed URL")
    title: Optional[str] = Field(None, examples=["Example Domain"], description="Website title")
    company_info: CompanyInfo = Field(..., description="Key extracted company information")
    questions_answered: Dict[str, str] = Field(...,
                                               examples=[{"What are the main products?": "Cloud computing services."}],
                                               description="Answers to the specific questions asked, based on website content")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow,
                                         description="Timestamp of the analysis completion (UTC)")
    processing_time_seconds: Optional[float] = Field(None, examples=[5.72],
                                                     description="Total time taken for scraping and analysis")
    success: bool = Field(True, description="Indicates if the analysis was successful")
    error_message: Optional[str] = Field(None, description="Error message if the analysis failed")
    ai_model_used: str = Field(default="rule-based", examples=["google-gemini", "rule-based (fallback)"],
                               description="AI model or method actually used for generating the primary company info and answers (e.g., 'google-gemini', 'rule-based', 'rule-based (fallback)')")


class ErrorDetail(BaseModel):
    error_code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = Field(False, description="Indicates failure")
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Security dependency
def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security, scopes=[])):
    """Verify API token."""
    if not SECRET_KEY:  # Server's key not set at all
        logger.critical(
            "API_SECRET_KEY is not configured on the server. Authentication is effectively disabled and will fail.")
        raise HTTPException(
            status_code=503,
            detail="API security not configured on server. Unable to authenticate.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Authorization header with Bearer token is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme. 'Bearer' scheme required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != SECRET_KEY:
        logger.warning(
            f"Invalid token attempt. Provided: '{credentials.credentials[:4]}...', Expected starts with: '{SECRET_KEY[:4]}...'")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(f"Token verified successfully for a client.")
    return credentials.credentials


# WebsiteScraper Class (remains largely the same, ensure robust extraction)
# ... (Keep your existing WebsiteScraper class here. For brevity, I'm omitting it but ensure it's present and correct)
# Make sure _extract_location prioritizes good signals for universities if needed.
class WebsiteScraper:  # Placeholder - use your full class
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 15

    def scrape_homepage(self, url: str) -> Dict[str, Any]:
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = 'https://' + url

            logger.info(f"Starting scrape for URL: {url}")
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            text_content = self._extract_text_content(soup)
            content_data = {
                'url': response.url,
                'title': self._extract_title(soup),
                'meta_description': self._extract_meta_description(soup),
                'headings': self._extract_headings(soup),
                'text_content': text_content,
                'contact_info': self._extract_contact_info(soup, response.text),
                'about_sections': self._extract_about_sections(soup),
                'status_code': response.status_code
            }
            content_data['location'] = self._extract_location(soup, text_content, content_data)

            logger.info(f"Successfully scraped content from: {response.url}")
            return content_data

        except requests.exceptions.Timeout:
            logger.error(f"Timeout error scraping {url}")
            raise HTTPException(status_code=408,
                                detail=f"Timeout: Failed to fetch website content from {url} within {self.timeout}s.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch website: {e}")
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to parse website content: {e}")

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None

    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '').strip() if meta_desc else None

    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, list]:
        headings = {}
        for i in range(1, 7):
            tag_name = f'h{i}'
            headings[tag_name] = [h.get_text(strip=True) for h in soup.find_all(tag_name)]
        return headings

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text[:10000]

    def _extract_contact_info(self, soup: BeautifulSoup, raw_text: str) -> Dict[str, Any]:
        contact_info = {'emails': [], 'phones': []}
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'  # Expanded TLD length
        contact_info['emails'] = list(set(re.findall(email_pattern, raw_text)))[:5]

        # Broader phone pattern trying to catch more international numbers; might need refinement
        phone_pattern = r'(?:\b\+?[0-9]{1,3}[-.\s]?)?\(?[0-9]{2,5}\)?[-.\s]?[0-9]{2,5}[-.\s]?[0-9]{2,5}(?:[-.\s]?[0-9]{1,5})?\b'
        phones = list(set(re.findall(phone_pattern, raw_text)))
        contact_info['phones'] = [p for p in phones if
                                  len(re.sub(r'\D', '', p)) >= 7 and len(re.sub(r'\D', '', p)) <= 15][:5]
        return contact_info

    def _extract_about_sections(self, soup: BeautifulSoup) -> list:
        about_sections = []
        about_keywords = ['about us', 'about-us', 'about reva', 'company', 'mission', 'vision', 'story', 'who we are',
                          'profile', 'overview']
        for keyword_phrase in about_keywords:
            # Search for text containing the keyword phrase, IDs, and classes
            elements = soup.find_all(string=re.compile(keyword_phrase, re.I))
            for el_text in elements:
                parent = el_text.parent
                if parent and parent.name in ['p', 'div', 'section', 'article']:  # Common content tags
                    text = parent.get_text(strip=True, separator=' ')
                    if text and 50 < len(text) < 2000:
                        about_sections.append(text[:500])
            # Also check classes/ids
            class_or_id_elements = soup.find_all(['div', 'section', 'article'],
                                                 attrs={'class': re.compile(keyword_phrase.replace(" ", "-"), re.I)},
                                                 limit=3)  # e.g. about-us
            class_or_id_elements.extend(soup.find_all(['div', 'section', 'article'],
                                                      attrs={'id': re.compile(keyword_phrase.replace(" ", "-"), re.I)},
                                                      limit=3))
            for el in class_or_id_elements:
                text = el.get_text(strip=True, separator=' ')
                if text and 50 < len(text) < 2000:
                    about_sections.append(text[:500])
        return list(set(about_sections))[:3]  # Unique sections, limit to 3

    def _extract_location(self, soup: BeautifulSoup, text_content: str, content_data: Dict[str, Any]) -> str:
        structured_location = self._extract_structured_location(soup)
        if structured_location and len(structured_location) > 3:
            logger.info(f"Extracted location from structured data: {structured_location}")
            return structured_location

        # Enhanced logic from your original file, slightly adapted
        title = content_data.get('title', '').lower()
        meta_desc = content_data.get('meta_description', '').lower()
        about_text = " ".join(content_data.get('about_sections', [])).lower()
        text_content_lower = text_content.lower()

        # Prioritize Bangalore/Karnataka for REVA if mentioned in title or meta
        if "reva" in title:  # Context specific for REVA
            if "bangalore" in title or "bengaluru" in title: return "Bangalore"
            if "karnataka" in title: return "Karnataka"

        # General search
        # Look for specific patterns first
        # Pattern for "City, State" or "City, Country"
        city_state_country_pattern = r'\b([A-Za-z\s]+?),\s*([A-Za-z\s]+)\b'
        # Simpler address like contexts
        address_context_keywords = ['address', 'contact us', 'located in', 'location', 'headquarters', 'visit us']

        # Combine text sources
        full_text_lower = f"{title} {meta_desc} {about_text} {text_content_lower}"

        # Known locations (prioritize specific ones for REVA if applicable)
        known_locations = [
            "Yelahanka, Bangalore", "Kattigenahalli, Bangalore", "Bangalore", "Bengaluru", "Karnataka", "India",
            # REVA specific
            "Delhi", "Mumbai", "Chennai", "Hyderabad", "Pune",  # Other Indian cities
            "London", "New York", "San Francisco", "California", "USA", "UK"  # International
        ]

        # Check for known locations in order of specificity
        for loc in known_locations:
            if loc.lower() in full_text_lower:
                # More weight if it's in title, meta, or about sections
                if loc.lower() in title or loc.lower() in meta_desc or loc.lower() in about_text:
                    logger.info(f"Found location '{loc}' in prominent page elements.")
                    return loc
                # Check if it's near address-like keywords
                for kw in address_context_keywords:
                    if kw in full_text_lower:
                        idx_kw = full_text_lower.find(kw)
                        idx_loc = full_text_lower.find(loc.lower())
                        if idx_loc != -1 and abs(idx_kw - idx_loc) < 200:  # If location is near keyword
                            logger.info(f"Found location '{loc}' near keyword '{kw}'.")
                            return loc

        # Fallback if no high-confidence known location is found near keywords
        for loc in known_locations:
            if loc.lower() in full_text_lower:
                logger.info(f"Found location '{loc}' in general text content.")
                return loc

        # Generic regex for City, ST type patterns if others fail
        matches = re.findall(city_state_country_pattern, full_text_lower)
        for city, region in matches:
            city = city.strip()
            region = region.strip()
            if len(city) > 2 and len(region) > 1 and not city.isdigit() and not region.isdigit():
                # Filter out common non-location phrases if necessary
                if city not in ["privacy policy", "terms of service"] and region not in ["privacy policy",
                                                                                         "terms of service"]:
                    found_loc_str = f"{city.title()}, {region.title()}"
                    logger.info(f"Extracted location by generic pattern: {found_loc_str}")
                    return found_loc_str

        logger.info("Location extraction did not find a definitive match, defaulting to Unknown.")
        return "Unknown"

    def _extract_structured_location(self, soup: BeautifulSoup) -> Optional[str]:
        # Schema.org for PostalAddress or Place
        for item_type in ["http://schema.org/PostalAddress", "http://schema.org/Place",
                          "https://schema.org/PostalAddress", "https://schema.org/Place"]:
            address_element = soup.find(attrs={"itemtype": item_type})
            if address_element:
                locality = address_element.find(attrs={"itemprop": "addressLocality"})
                region = address_element.find(attrs={"itemprop": "addressRegion"})
                country = address_element.find(attrs={"itemprop": "addressCountry"})
                street = address_element.find(attrs={"itemprop": "streetAddress"})

                parts = []
                if street and street.get_text(strip=True): parts.append(street.get_text(strip=True))
                if locality and locality.get_text(strip=True): parts.append(locality.get_text(strip=True))
                if region and region.get_text(strip=True): parts.append(region.get_text(strip=True))
                if country and country.get_text(strip=True): parts.append(country.get_text(strip=True))

                if parts: return ", ".join(parts)

        # Geo meta tags
        geo_placename = soup.find("meta", attrs={"name": "geo.placename"})
        if geo_placename and geo_placename.get("content"): return geo_placename["content"]
        geo_region = soup.find("meta", attrs={"name": "geo.region"})  # e.g. US-CA
        if geo_region and geo_region.get("content"): return geo_region["content"]

        return None


# Google Gemini AI Analyzer Class
class GeminiAnalyzer:
    def __init__(self):
        self.model = None
        if ENABLE_AI and 'genai' in globals() and GOOGLE_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("GeminiAnalyzer initialized with gemini-2.0-flash model.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}. AI analysis will fallback to rules.",
                             exc_info=True)
                self.model = None
        elif ENABLE_AI and not GOOGLE_API_KEY:
            logger.warning(
                "Gemini AI is enabled but GOOGLE_API_KEY is missing. AI analysis will use rule-based fallback.")
        elif not ENABLE_AI:
            logger.info("Gemini AI is disabled by configuration. Using rule-based analysis.")

    async def analyze_content(self, content_data: Dict[str, Any], questions: List[str]) -> Tuple[Dict[str, Any], str]:
        """
        Analyzes content using Gemini AI or falls back to rule-based analysis.
        Returns a tuple: (analysis_result_dict, method_used_str)
        """
        method_used = "rule-based (initial)"  # Default if AI not attempted
        if self.model and ENABLE_AI:  # Attempt AI only if model is available and AI is enabled
            try:
                analysis_text = self._prepare_content_for_ai(content_data)
                prompt = self._create_analysis_prompt(analysis_text, questions, content_data.get('url', 'the website'))

                logger.info(f"Sending request to Gemini API for URL: {content_data.get('url')}")
                response_text = await asyncio.get_event_loop().run_in_executor(
                    executor, self._call_gemini_api, prompt
                )
                logger.info(f"Received response from Gemini API for URL: {content_data.get('url')}")

                parsed_result = self._parse_ai_response(response_text, questions)
                # If AI parsing itself signals an error, we might still say AI was attempted but failed at parsing
                if "Error parsing AI response" in parsed_result.get("industry", ""):
                    method_used = "google-gemini (parsing_failed_fallback_to_rules)"
                    # Potentially re-run rule-based here or ensure parse_ai_response gives a clean rule-like fallback
                    logger.warning(
                        f"Gemini response parsing failed for {content_data.get('url')}. Result may be incomplete or rule-based.")
                    # To ensure consistency if parsing fails badly, explicitly use rule-based output
                    rule_based_result = await self._rule_based_analysis(content_data, questions,
                                                                        error_source="AI Response Parsing Error")
                    return rule_based_result, method_used

                return parsed_result, "google-gemini"

            except Exception as e:
                logger.error(
                    f"Gemini analysis failed for URL {content_data.get('url')}: {e}. Falling back to rule-based analysis.",
                    exc_info=True)
                method_used = "rule-based (AI_exception_fallback)"
                # Fall through to rule-based analysis below

        # If AI was not enabled, model is None, or an exception occurred above:
        logger.info(
            f"Using rule-based analysis for {content_data.get('url')}. Reason: {method_used if 'fallback' in method_used else 'AI not enabled/configured'}")
        rule_based_output = await self._rule_based_analysis(content_data, questions, error_source=method_used)
        return rule_based_output, method_used  # Use the more specific method_used if it was an AI fallback

    def _prepare_content_for_ai(self, content_data: Dict[str, Any]) -> str:
        # ... (same as your previous _prepare_content_for_ai, ensure it's concise and relevant)
        content_parts = [
            f"Website URL: {content_data.get('url', 'N/A')}",
            f"Website Title: {content_data.get('title', 'N/A')}",
            f"Meta Description: {content_data.get('meta_description', 'N/A')}",
        ]
        if content_data.get('headings'):
            h1s = content_data['headings'].get('h1', [])
            if h1s: content_parts.append(f"Main Headings (H1): {'; '.join(h1s[:3])}")
            h2s = content_data['headings'].get('h2', [])
            if h2s: content_parts.append(f"Sub Headings (H2): {'; '.join(h2s[:3])}")

        about_sections_text = ' '.join(content_data.get('about_sections', []))
        if about_sections_text:
            content_parts.append(f"About/Company Sections: {about_sections_text[:1500]}")  # Limit length

        text_to_analyze = content_data.get('text_content', '')
        if len(text_to_analyze) > 7000:
            text_to_analyze = text_to_analyze[:7000]  # Further reduced for very dense pages
        content_parts.append(f"\nKey Website Content Snippets:\n{text_to_analyze}")

        return '\n\n'.join(filter(None, content_parts))

    def _create_analysis_prompt(self, content: str, questions: List[str], site_url_for_context: str) -> str:
        questions_formatted = "\n".join([f"- \"{q}\"" for q in questions])

        # Adding specific instruction for educational institutions if site_url_for_context hints at it
        entity_type_hint = "company or organization"
        if ".edu" in site_url_for_context or "university" in site_url_for_context.lower() or "college" in site_url_for_context.lower():
            entity_type_hint = "educational institution (like a university or college)"

        return f"""
Analyze the following website content for an {entity_type_hint}.
Based ONLY on the provided website content, answer the specified questions.
If information is not clearly available in the content, explicitly state "Unknown" or "Not specified in the provided content". Do not invent or assume information.

Website Content:
------------------
{content}
------------------

Based strictly on the content above, provide the following information in JSON format:

1.  **Overall Information**:
    * `industry`: The specific industry or sector (e.g., "Higher Education", "Software Development", "E-commerce Retail"). For educational sites, use "Education" or more specific like "Higher Education".
    * `company_size`: An estimated size. For companies: "Startup", "Small Business (1-50 employees)", "Medium Business (51-500 employees)", "Large Enterprise (501+ employees)". For educational institutions: "Small Institution", "Medium Institution", "Large Institution" based on typical university/college scales if inferable, otherwise "Unknown".
    * `location`: The primary physical location or headquarters (e.g., "London, UK", "Remote", "Bengaluru, India").
    * `description`: A brief (1-2 sentences) objective description of what the {entity_type_hint} does or its main purpose.
    * `confidence_score`: Your confidence (0.0 to 1.0) in the accuracy of these four fields, based on how explicit the information is in the content.

2.  **Answers to Specific Questions**:
    Provide answers for each of the following questions. Map the original question text to its answer.
{questions_formatted}

Respond with ONLY a single valid JSON object. Do not include any explanatory text before or after the JSON.
Example for questions_answered:
"questions_answered": {{
    "What industry does this company belong to?": "Higher Education",
    "What is the approximate size of the company?": "Large Institution"
}}

JSON Structure Template:
{{
    "industry": "...",
    "company_size": "...",
    "location": "...",
    "description": "...",
    "confidence_score": 0.0,
    "questions_answered": {{
        "Original Question 1 Text": "Answer to question 1...",
        "Original Question 2 Text": "Answer to question 2..."
    }}
}}
"""

    # _call_gemini_api and _parse_ai_response remain the same as your last provided app.py
    # Ensure _parse_ai_response handles the structure from _create_analysis_prompt.
    def _call_gemini_api(self, prompt: str) -> str:
        """Calls Google Gemini API and returns the text response."""
        if not self.model:
            raise Exception("Gemini model not initialized.")
        try:
            logger.info("Attempting to generate content with Gemini API.")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=2048,
                    temperature=0.15,  # Even lower for more factual
                )
            )
            if response.parts:
                full_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                if full_text:
                    logger.info("Successfully received text from Gemini API.")
                    return full_text
            if hasattr(response, 'text') and response.text:
                logger.info("Successfully received simple text from Gemini API.")
                return response.text

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                logger.error(f"Gemini API call blocked. Reason: {block_reason}")
                raise Exception(f"Gemini API content generation blocked: {block_reason}")
            else:
                logger.error("Gemini API returned no text content and no explicit block reason.")
                raise Exception("Gemini API returned no text content.")

        except (getattr(globals().get('genai', {}), 'types', {}).generation_types.BlockedPromptException,
                getattr(globals().get('google', {}).api_core.exceptions, 'GoogleAPIError')) as e:
            logger.error(f"Gemini API call failed: {type(e).__name__} - {str(e)}", exc_info=True)
            if hasattr(e, 'message') and "API key not valid" in e.message:
                raise Exception(f"Gemini API Key is invalid. Please check your GOOGLE_API_KEY. Details: {str(e)}")
            if isinstance(e, getattr(globals().get('google', {}).api_core.exceptions, 'PermissionDenied', type(None))):
                raise Exception(
                    f"Gemini API Permission Denied. Check API key and ensure the Generative Language API is enabled. Details: {str(e)}")
            raise Exception(f"Gemini API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Gemini API call: {type(e).__name__} - {str(e)}", exc_info=True)
            raise

    def _parse_ai_response(self, response_text: str, questions: List[str]) -> Dict[str, Any]:
        try:
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[len("```json"):]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-len("```")]
            cleaned_response_text = cleaned_response_text.strip()

            if not cleaned_response_text:
                logger.error("AI response was empty after cleaning.")
                raise json.JSONDecodeError("Empty response string", cleaned_response_text, 0)

            parsed_json = json.loads(cleaned_response_text)

            company_info_fields = {
                "industry": parsed_json.get('industry', "Unknown"),
                "company_size": parsed_json.get('company_size', "Unknown"),
                "location": parsed_json.get('location', "Unknown"),
                "description": parsed_json.get('description', "Not specified"),
                "confidence_score": parsed_json.get('confidence_score', 0.5)
            }

            answered_questions_from_ai = parsed_json.get('questions_answered', {})
            final_answers = {}
            for q_text in questions:
                final_answers[q_text] = answered_questions_from_ai.get(q_text,
                                                                       "AI did not provide a specific answer for this question.")

            return {**company_info_fields, "questions_answered": final_answers}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}. Raw response snippet: '{response_text[:500]}...'")
            return {  # Fallback structure mirrors expected fields but flags error
                "industry": "Error parsing AI response", "company_size": "Error parsing AI response",
                "location": "Error parsing AI response",
                "description": "AI response could not be parsed into the expected JSON format.",
                "confidence_score": 0.1,
                "questions_answered": {q: "Error: Could not parse AI analysis results." for q in questions}
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {e}. Raw response snippet: '{response_text[:500]}...'")
            return {
                "industry": "Unknown (Parsing Error)", "company_size": "Unknown (Parsing Error)",
                "location": "Unknown (Parsing Error)",
                "description": "Analysis unavailable due to an unexpected error during AI response parsing.",
                "confidence_score": 0.1,
                "questions_answered": {q: "Unable to analyze due to unexpected AI response parsing error." for q in
                                       questions}
            }

    async def _rule_based_analysis(self, content_data: Dict[str, Any], questions: List[str],
                                   error_source: Optional[str] = None) -> Dict[str, Any]:
        reason = error_source if error_source else 'AI not enabled or unavailable'
        logger.info(f"Using rule-based analysis. Reason: {reason}")

        text_content_lower = (
                str(content_data.get('title', '')) + ' ' +
                str(content_data.get('meta_description', '')) + ' ' +
                ' '.join(content_data.get('about_sections', [])) + ' ' +
                str(content_data.get('text_content', ''))
        ).lower()
        url_lower = str(content_data.get('url', '')).lower()

        def find_keyword(keywords: List[str], text: str) -> bool:
            return any(keyword in text for keyword in keywords)

        # Improved industry map, prioritizing Education
        industry_map = [  # List of tuples for ordered checking
            ("Higher Education",
             ["university", "college", "academic institute", "higher education", "b.tech", "m.tech", "ph.d",
              "undergraduate", "postgraduate"]),
            ("Education", ["school", "education", "learning", "courses", "students", "faculty", "campus"]),
            ("Technology/Software",
             ["software", "tech", "cloud", "app", "ai", "data", "platform", "saas", "it services"]),
            ("E-commerce/Retail",
             ["shop", "store", "sell", "buy", "market", "retail", "e-commerce", "online shopping"]),
            ("Healthcare", ["health", "medical", "clinic", "hospital", "pharma", "patient care"]),
            ("Finance", ["finance", "bank", "invest", "payment", "insurance", "fintech"]),
            ("Consulting", ["consulting", "services", "advisory", "solutions", "business strategy"]),
            ("Manufacturing", ["manufacturing", "industrial", "factory", "production"]),
        ]
        detected_industry = "Unknown"
        if ".edu" in url_lower:  # Strong hint for education
            detected_industry = "Education"  # Default for .edu, can be refined by keywords
        for industry, kws in industry_map:
            if find_keyword(kws, text_content_lower):
                detected_industry = industry
                break  # Take first match from prioritized list

        # Size detection
        size_map = {  # General company sizes
            "Large": ["fortune 500", "global", "worldwide", "multinational", "enterprise", "corporation",
                      " thousands of employees"],
            "Medium": ["growing", "expanding", "regional", "established", "hundreds of employees"],
            "Small": ["startup", "small business", "local", " tens of employees", " boutique"],
        }
        detected_size = "Unknown"
        # Specific for education
        if detected_industry in ["Education", "Higher Education"]:
            if find_keyword(["university", "renowned", "established since", "multiple faculties", "large campus"],
                            text_content_lower):
                detected_size = "Large (Educational Institution)"
            elif find_keyword(["college", "institute", "several departments"], text_content_lower):
                detected_size = "Medium (Educational Institution)"
            elif find_keyword(["training center", "small institute"], text_content_lower):
                detected_size = "Small (Educational Institution)"
            else:  # Default for general .edu if no other size cues
                if ".edu" in url_lower: detected_size = "Educational Institution (Size Unspecified)"

        else:  # For other industries
            for size, indicators in size_map.items():
                if find_keyword(indicators, text_content_lower):
                    detected_size = size
                    break

        detected_location = content_data.get('location', "Unknown")

        description = f"Rule-based analysis suggests an entity in the {detected_industry} sector."
        if detected_industry == "Unknown":
            description = "Rule-based analysis could not determine a specific industry."

        answers = {}
        for q in questions:
            q_lower = q.lower()
            if "industry" in q_lower:
                answers[q] = detected_industry
            elif "size" in q_lower:
                answers[q] = detected_size
            elif "location" in q_lower or "where" in q_lower:
                answers[q] = detected_location
            else:
                answers[q] = "Information for this question could not be determined by this rule-based analysis."

        return {
            "industry": detected_industry,
            "company_size": detected_size,
            "location": detected_location,
            "description": description,
            "confidence_score": 0.45,  # Slightly higher for improved rules
            "questions_answered": answers
        }


# Initialize components
scraper = WebsiteScraper()
analyzer = GeminiAnalyzer()


# API Endpoints
@app.get("/", tags=["Health"], summary="Root health check")
# ... (Root endpoint as before)
async def root():
    return {
        "message": "AI Website Business Intelligence API (Powered by Google Gemini) is running.",
        "status": "healthy",
        "version": app.version,
        "ai_enabled": ENABLE_AI,
        "ai_provider": "Google Gemini" if ENABLE_AI and GOOGLE_API_KEY and analyzer.model else "None (Rule-based fallback or AI not configured)",
        "timestamp": datetime.utcnow()
    }


@app.get("/health", tags=["Health"], summary="Detailed health check")
# ... (Health endpoint as before, ensure gemini_status reflects analyzer.model)
async def health_check():
    gemini_status = "disabled_by_config"
    if ENABLE_AI:  # This flag itself is now more reliable
        if GOOGLE_API_KEY and analyzer.model:
            gemini_status = "operational"
        elif GOOGLE_API_KEY and not analyzer.model:
            gemini_status = "initialization_error (check API key/library)"
        elif not GOOGLE_API_KEY:
            gemini_status = "api_key_missing"

    auth_properly_configured = bool(SECRET_KEY)  # Basic check if server secret is set

    return {
        "overall_status": "healthy" if auth_properly_configured else "degraded_security_key_missing_on_server",
        "timestamp": datetime.utcnow(),
        "version": app.version,
        "authentication_status": "Server API_SECRET_KEY is SET" if auth_properly_configured else "CRITICAL: Server API_SECRET_KEY is NOT SET in .env",
        "components": {
            "api_server": "operational",
            "website_scraper": "operational",
            "ai_analyzer": {
                "status": gemini_status,
                "provider": "Google Gemini" if ENABLE_AI else "None"
            }
        },
        "ai_enabled_flag_effective": ENABLE_AI  # Reflects if AI can actually run
    }


@app.post("/analyze",
          response_model=WebsiteAnalysisResponse,
          # ... (responses as before)
          tags=["Analysis"],
          summary="Analyze a single website")
async def analyze_website(
        request_data: WebsiteAnalysisRequest,
        _token: str = Depends(verify_token)
):
    start_time = datetime.utcnow()
    logger.info(f"Authenticated analysis request for URL: {request_data.url}")

    try:
        content_data = scraper.scrape_homepage(str(request_data.url))

        # analyze_content now returns a tuple: (dict_result, str_method_used)
        analysis_output_dict, actual_method_used = await analyzer.analyze_content(
            content_data, request_data.questions
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        company_info_obj = CompanyInfo(
            industry=analysis_output_dict.get('industry'),
            company_size=analysis_output_dict.get('company_size'),
            location=analysis_output_dict.get('location'),
            description=analysis_output_dict.get('description'),
            confidence_score=analysis_output_dict.get('confidence_score')
        )

        response = WebsiteAnalysisResponse(
            url=str(request_data.url),
            title=content_data.get('title'),
            company_info=company_info_obj,
            questions_answered=analysis_output_dict.get('questions_answered', {}),
            processing_time_seconds=round(processing_time, 2),
            ai_model_used=actual_method_used,  # Use the method returned by analyze_content
            success=True  # Assuming if we reach here, it's a structural success
        )
        # Check if the analysis itself (e.g. parsing AI) flagged an issue internally
        if "Error" in str(analysis_output_dict.get('industry', '')) or "Error" in str(
                analysis_output_dict.get('description', '')):
            logger.warning(
                f"Analysis for {request_data.url} completed but with internal processing issues. Method: {actual_method_used}")
            # success flag remains true as the endpoint itself worked, but data might reflect errors.
        else:
            logger.info(
                f"Analysis successful for {request_data.url}. Time: {processing_time:.2f}s. Actual Method: {actual_method_used}")
        return response

    except HTTPException as he:
        logger.warning(f"HTTPException during analysis for {request_data.url}: {he.status_code} - {he.detail}",
                       exc_info=True)
        raise
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"Unexpected critical error analyzing {request_data.url}: {e}", exc_info=True)
        return WebsiteAnalysisResponse(
            url=str(request_data.url),
            title=None,
            company_info=CompanyInfo(description=f"Critical error during analysis: {type(e).__name__}"),
            questions_answered={q: "Analysis failed due to an internal server error." for q in request_data.questions},
            processing_time_seconds=round(processing_time, 2),
            success=False,
            error_message=f"An unexpected internal error occurred: {str(e)}",
            ai_model_used="error_in_processing"
        )


# Custom Exception Handlers (remain the same)
# ... (Keep your existing custom_http_exception_handler and generic_exception_handler)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.info(f"HTTPException caught by handler: {exc.status_code} - {exc.detail}. Path: {request.url.path}")
    return json_response(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(error_code=str(exc.status_code), message=exc.detail),
        ).model_dump(),
        headers=exc.headers if hasattr(exc, 'headers') else None
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled generic exception by handler: {type(exc).__name__} - {exc}. Path: {request.url.path}",
                 exc_info=True)
    return json_response(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(error_code="500", message="An unexpected internal server error occurred."),
        ).model_dump()
    )


from fastapi.responses import JSONResponse as json_response


# Startup and Shutdown events (remain largely the same, check logging messages)
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI AI Website Business Intelligence API is starting up...")
    if not SECRET_KEY:
        logger.critical(
            "CRITICAL SECURITY WARNING: API_SECRET_KEY is NOT SET in the environment. Authentication for /analyze WILL FAIL.")
    else:
        logger.info(f"API_SECRET_KEY loaded, starting with '{SECRET_KEY[:4]}...'.")

    logger.info(f"Configured AI Enabled (ENABLE_AI env var): {os.getenv('ENABLE_AI', 'false').lower() == 'true'}")
    logger.info(f"Effective AI Status (after checks): {'Enabled' if ENABLE_AI else 'Disabled'}")
    if ENABLE_AI:
        if not GOOGLE_API_KEY:
            logger.warning(
                "Google Gemini AI is marked enabled, but GOOGLE_API_KEY is missing. AI features will not function.")
        elif not analyzer.model:
            logger.warning(
                "Google Gemini AI is marked enabled, and key is present, but the model failed to initialize. AI features will not function.")
        else:
            logger.info("Google Gemini AI provider configured and model appears initialized.")
    else:
        logger.info("AI analysis is disabled. Using rule-based fallback only.")
    logger.info("Application startup complete.")


@app.on_event("shutdown")
# ... (Shutdown event as before)
async def shutdown_event():
    logger.info("FastAPI AI Website Business Intelligence API is shutting down...")
    executor.shutdown(wait=True)
    logger.info("Thread pool shut down. Application shutdown complete.")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server directly from app.py (for development/debugging)")

    app_host = os.getenv("HOST", "127.0.0.1")
    app_port = int(os.getenv("PORT", 8000))
    app_reload = os.getenv("DEBUG", "true").lower() == "true"

    uvicorn.run(
        "app:app",
        host=app_host,
        port=app_port,
        reload=app_reload,
        log_level="info"
    )