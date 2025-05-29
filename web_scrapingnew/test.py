#!/usr/bin/env python3
"""
Test client for the FastAPI AI Website Business Intelligence API.
Prompts for API Key if not found in arguments.

Usage:
  python test.py <url_to_analyze> [api_key]
  python test.py (interactive mode)
"""

import requests
import json
import sys
from datetime import datetime

class WebsiteAnalyzerClient:
    """Client for testing the website analyzer API."""

    def __init__(self, base_url="http://127.0.0.1:8000", api_key_arg=None):
        self.base_url = base_url.rstrip('/')

        # Always require API key: from argument or prompt
        if api_key_arg is not None:
            determined_api_key = api_key_arg
            print(f"ℹ️  Using API key from command-line argument: '{determined_api_key[:4]}...'")
        elif sys.stdin.isatty():
            try:
                api_key_input = input("🔑 Enter API Secret Key for authorization (required for /analyze): ").strip()
                if api_key_input:
                    determined_api_key = api_key_input
                    print("✅ API key received from prompt.")
                else:
                    print("❌ No API key entered. Exiting.")
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\n🛑 API Key input cancelled by user. Exiting.")
                sys.exit(1)
        else:
            print("❌ No API key provided and not running in interactive mode. Exiting.")
            sys.exit(1)

        self.api_key = determined_api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(f"ℹ️  Client will use API Key for authenticated requests.")

    def _make_request(self, method, endpoint, **kwargs):
        try:
            url = f"{self.base_url}{endpoint}"
            effective_headers = self.headers.copy()
            if self.api_key:
                effective_headers["Authorization"] = f"Bearer {self.api_key}"
            elif "Authorization" in effective_headers:
                del effective_headers["Authorization"]

            response = requests.request(method, url, headers=effective_headers, timeout=90, **kwargs)
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"error": "Non-JSON response from server", "status_code": response.status_code, "text_preview": response.text[:200]}
            return response.status_code, response_data
        except requests.exceptions.Timeout:
            return 408, {"error": "Client request timed out", "detail": f"The request to {url} timed out."}
        except requests.exceptions.ConnectionError:
            return 503, {"error": "Client connection error", "detail": f"Could not connect to {self.base_url}. Is the server running?"}
        except Exception as e:
            return 500, {"error": "Client-side exception during request", "detail": str(e)}

    def analyze_website(self, url_to_analyze: str, questions: list = None):
        payload = {"url": url_to_analyze}
        if questions:
            payload["questions"] = questions

        print(f"\n--- Attempting to Analyze Website: {url_to_analyze} ---")
        if not self.api_key:
            print("   (Client has no API key; expecting server to handle authentication if endpoint is protected)")
        if questions:
            print(f"   With questions: {questions}")

        start_time = datetime.now()
        status_code, data = self._make_request("POST", "/analyze", json=payload)
        duration = (datetime.now() - start_time).total_seconds()
        return status_code, data, duration

    def print_simple_response(self, name, status_code, data):
        print(f"📦 Response from {name}: Status Code {status_code}")
        try:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except TypeError:
            print(str(data))
        if status_code >= 400:
            print(f"🔥 Received HTTP error for {name}!")

    def print_analysis_results(self, url, status_code, result_data, duration):
        print("-" * 70)
        print(f"📊 Analysis Results for: {url}")
        print(f"⏱️  Client Request Duration: {duration:.2f} seconds")
        print(f"🚦 HTTP Status Code from Server: {status_code}")

        if status_code == 200:
            if result_data.get('success', False):
                print("✅ Analysis Successful (as reported by server)!")
                print(f"   🔗 Analyzed URL: {result_data.get('url')}")
                print(f"   📰 Website Title: {result_data.get('title', 'N/A')}")
                print(f"   ⏱️  Server Processing Time: {result_data.get('processing_time_seconds', 'N/A')}s")
                print(f"   🤖 AI Model Used: {result_data.get('ai_model_used', 'N/A')}")
                company_info = result_data.get('company_info', {})
                print("\n   --- Company Information (from server) ---")
                print(f"      🏭 Industry: {company_info.get('industry', 'Unknown')}")
                print(f"      📏 Size: {company_info.get('company_size', 'Unknown')}")
                print(f"      📍 Location: {company_info.get('location', 'Unknown')}")
                print(f"      📝 Description: {company_info.get('description', 'N/A')}")
                print(f"      🎯 Confidence: {company_info.get('confidence_score', 'N/A')}")
                questions_answered = result_data.get('questions_answered', {})
                if questions_answered:
                    print("\n   --- Answers to Your Questions (from server) ---")
                    for i, (question, answer) in enumerate(questions_answered.items(), 1):
                        print(f"      {i}. Q: {question}")
                        print(f"         A: {answer}")
                else:
                    print("\n   No specific question answers provided in this format by server.")
            else:
                print("⚠️ Server responded with HTTP 200, but indicated failure in response payload:")
                print(f"   Message: {result_data.get('error_message', 'No specific error message from server.')}")
                print(f"   Details: {json.dumps(result_data, indent=2, ensure_ascii=False)}")
        else:
            print("❌ Analysis Failed - Server Returned HTTP Error!")
            print("   Server Error Details:")
            print(json.dumps(result_data, indent=2, ensure_ascii=False))
            error_detail_obj = result_data.get("error", {})
            error_message_from_payload = ""
            if isinstance(error_detail_obj, dict):
                error_message_from_payload = error_detail_obj.get("message", "")
            elif isinstance(result_data.get("detail"), str):
                error_message_from_payload = result_data.get("detail", "")
            if status_code == 401:
                print("\n   💡 AUTHENTICATION FAILED (401):")
                if "Invalid authentication credentials" in error_message_from_payload:
                    print("      Server rejected the API Key. Ensure the key entered/used matches the one in the server's .env file.")
                elif "Not authenticated" in error_message_from_payload or "missing" in error_message_from_payload.lower():
                    print("      Server indicates no API Key (Bearer token) was received or it was malformed. Ensure the client sent the key correctly.")
                else:
                    print(f"      Server detail: {error_message_from_payload if error_message_from_payload else 'No specific detail in error message.'}")
            elif status_code == 403:
                print("\n   💡 AUTHORIZATION FORBIDDEN (403):")
                print(f"      Server detail: {error_message_from_payload if error_message_from_payload else 'No specific detail.'}")
            elif status_code == 503 and "API security not configured" in error_message_from_payload:
                print("\n   💡 SERVICE UNAVAILABLE (503): The server reported it's not configured with an API_SECRET_KEY for authentication.")
                print("      Check the server's .env file and restart it.")
            elif result_data.get("error") == "Non-JSON response from server":
                print("\n   💡 UNEXPECTED SERVER RESPONSE: The server did not return valid JSON. This might indicate a server-side crash or misconfiguration.")
                print(f"      Status Code: {result_data.get('status_code')}, Preview: '{result_data.get('text_preview')}'")
        print("-" * 70)

def run_interactive_mode(client: WebsiteAnalyzerClient):
    print("\n🔧 Interactive Mode: Enter website URLs to analyze.")
    print("   Type 'custom' to also enter custom questions for the next URL.")
    print("   Type 'quit' or 'exit' to stop.")
    custom_questions_next = False
    default_questions = [
        "What are the main products or services offered by this company?",
        "Who is the target audience or customer base for this website?",
        "What is the primary call to action on the homepage?"
    ]
    while True:
        try:
            url_input = input("\n🌐 Enter URL (or 'custom', 'quit'): ").strip()
            if url_input.lower() in ['quit', 'exit', 'q']:
                break
            if not url_input:
                continue
            if url_input.lower() == 'custom':
                custom_questions_next = True
                print("   Next URL will use custom questions.")
                continue
            if not url_input.startswith(('http://', 'https://')):
                url_to_analyze = 'https://' + url_input
            else:
                url_to_analyze = url_input
            questions_for_url = None
            if custom_questions_next:
                print(f"   Enter custom questions for {url_to_analyze} (one per line, empty line to finish):")
                questions_for_url = []
                while True:
                    q = input("      Question: ").strip()
                    if not q: break
                    questions_for_url.append(q)
                custom_questions_next = False
            else:
                use_defaults_input = input(f"   Use default questions for {url_to_analyze}? (Y/n/custom): ").strip().lower()
                if use_defaults_input == 'n':
                    questions_for_url = []
                elif use_defaults_input == 'custom':
                    custom_questions_next = True
                    print("   Re-enter URL after specifying it's for custom questions.")
                    continue
                else:
                    questions_for_url = default_questions
            status, data, duration = client.analyze_website(url_to_analyze, questions_for_url)
            client.print_analysis_results(url_to_analyze, status, data, duration)
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred in the client: {e}")

def main():
    print("🤖 FastAPI AI Website Scraper - Test Client")
    print("=" * 70)
    api_key_arg = None
    url_arg = None
    if len(sys.argv) > 1:
        if not sys.argv[1].startswith("-"):
            url_arg = sys.argv[1]
        if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
            api_key_arg = sys.argv[2]
        if "--help" in sys.argv or "-h" in sys.argv:
            print("Usage: python test.py [url_to_analyze] [api_key]")
            print("If no arguments are provided, enters interactive mode.")
            sys.exit(0)
    client = WebsiteAnalyzerClient(api_key_arg=api_key_arg)

    # Immediately check API key validity by calling /analyze with a dummy URL
    print("\n--- Checking API Key validity (/analyze) ---")
    dummy_url = "https://www.reva.edu.in/"
    status_code, data, duration = client.analyze_website(dummy_url, questions=[
        "What industry does this company belong to?"
    ])
    if status_code == 401:
        print("❌ Authentication failed: Invalid API key. Exiting.")
        sys.exit(1)
    elif status_code >= 400:
        print(f"⚠️  Server returned error code {status_code} during API key check.")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        sys.exit(1)
    else:
        print("✅ API key is valid.")

    if url_arg:
        print(f"\n🎯 Analyzing single URL from argument: {url_arg}")
        questions = [
            "What industry does this company belong to?",
            "What is the approximate size of the company?",
            "Where is the company located?"
        ]
        status, data, duration = client.analyze_website(url_arg, questions)
        client.print_analysis_results(url_arg, status, data, duration)
    else:
        if sys.stdin.isatty():
            run_interactive_mode(client)
        else:
            print("ℹ️ No URL provided as argument and not in interactive TTY mode. Exiting.")
            print("   Run with a URL: python test.py <url_to_analyze> [api_key]")
            print("   Or run interactively: python test.py")

if __name__ == "__main__":
    main()
    print("\n👋 Test client finished.")