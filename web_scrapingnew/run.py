#!/usr/bin/env python3
"""
Local development server startup script for the FastAPI AI Website Scraper.
Ensures environment variables from .env are loaded before starting Uvicorn.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv  # Import python-dotenv


def check_python_requirements():
    """Check if essential Python packages for running the app seem to be installed."""
    required_packages = ["fastapi", "uvicorn", "requests", "beautifulsoup4", "pydantic", "google-generativeai",
                         "python-dotenv"]
    missing = []
    try:
        for package_name in required_packages:
            # A simple import check; for more robustness, could check specific versions.
            __import__(package_name.replace("-",
                                            "_"))  # e.g. beautifulsoup4 -> bs4, google-generativeai -> google.generativeai
    except ImportError as e:
        # This specific check is tricky due to package name vs import name.
        # A requirements.txt check is usually more reliable.
        # For now, this is a basic check.
        pass  # Simplistic check, better to rely on pip install -r requirements.txt

    # A better check would be to ensure requirements.txt is installed.
    # This is a placeholder for a more robust check if needed.
    print("ℹ️  Please ensure all packages from requirements.txt are installed (`pip install -r requirements.txt`).")
    return True


def verify_env_file_and_keys():
    """Checks for .env file and essential keys, guiding the user if necessary."""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found.")
        print("   Attempting to create one from .env.example if it exists...")
        example_env_path = Path(".env.example")
        if example_env_path.exists():
            import shutil
            shutil.copy(example_env_path, env_path)
            print(
                f"✅ Created .env from {example_env_path}. Please review and fill in your details (especially API keys).")
        else:
            print(
                f"   Could not find .env.example. Please create a .env file with required variables like API_SECRET_KEY and GOOGLE_API_KEY.")
        return False  # Indicates setup might be incomplete

    # After loading, check if essential keys are present and not default
    api_secret_key = os.getenv("API_SECRET_KEY")
    enable_ai = os.getenv("ENABLE_AI", "false").lower() == "true"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    all_good = True
    if not api_secret_key or api_secret_key == "your-secret-key-here":  # Check against a common placeholder
        print(
            "❌ CRITICAL: API_SECRET_KEY is missing or using a default placeholder in .env. Please set a strong, unique key.")
        all_good = False

    if enable_ai and (not google_api_key or google_api_key == "your-google-api-key-here"):
        print(
            "❌ WARNING: ENABLE_AI is true, but GOOGLE_API_KEY is missing or using a default placeholder in .env. AI features may not work.")
        # This might not be critical if rule-based fallback is acceptable, but good to warn.
        # all_good = False # Uncomment if AI is strictly required when enabled

    if all_good:
        print("✅ .env file processed and essential keys seem to be configured.")
    else:
        print("   Please update your .env file accordingly.")
    return all_good


def main():
    """Main function to start the local development server."""
    print("🚀 Starting FastAPI AI Website Business Intelligence Server...")
    print("==================================================================")

    # Load environment variables from .env file into os.environ
    # This makes them available to this script and the Uvicorn process.
    if load_dotenv():
        print("✅ Environment variables loaded from .env file.")
    else:
        print("ℹ️  No .env file found or it's empty. Relying on system environment variables.")

    # Perform checks after attempting to load .env
    if not check_python_requirements():
        print("❌ Halting due to missing Python requirements.")
        sys.exit(1)

    if not verify_env_file_and_keys():
        print("❌ Environment configuration is incomplete or insecure. Please address the issues above.")
        print("   Server will attempt to start, but may not function correctly or securely.")
        # sys.exit(1) # Or choose to exit if config is critical

    # Get Uvicorn configuration from environment variables (with defaults)
    host = os.getenv("HOST", "127.0.0.1")  # Default to 127.0.0.1 for local security
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))  # Number of Uvicorn workers
    reload_dev = os.getenv("DEBUG", "true").lower() == "true"  # Enable reload if DEBUG=true

    print("\n📋 Server Configuration:")
    print(f"   - Host: {host}")
    print(f"   - Port: {port}")
    print(f"   - Workers: {workers if not reload_dev else 1}")  # Reload mode typically uses 1 worker
    print(f"   - Debug/Reload Mode: {'Enabled' if reload_dev else 'Disabled'}")

    secret_key_status = "Set" if os.getenv("API_SECRET_KEY") and os.getenv(
        "API_SECRET_KEY") != "your-secret-key-here" else "MISSING or Default"
    print(f"   - API Secret Key: {secret_key_status}")

    ai_enabled_status = os.getenv("ENABLE_AI", "false").lower()
    print(f"   - AI Enabled: {ai_enabled_status}")
    if ai_enabled_status == "true":
        google_key_status = "Set" if os.getenv("GOOGLE_API_KEY") and os.getenv(
            "GOOGLE_API_KEY") != "your-google-api-key-here" else "MISSING or Default"
        print(f"     - Google Gemini API Key: {google_key_status}")

    print(f"\n🔗 API Documentation (Swagger UI): http://{host}:{port}/docs")
    print(f"🔗 Alternative API Docs (ReDoc): http://{host}:{port}/redoc")
    print(f"🩺 Health Check: http://{host}:{port}/health")

    print("\n🔧 Press Ctrl+C to stop the server.")
    print("==================================================================")

    # Start Uvicorn server using subprocess to ensure it's a separate process
    # or use uvicorn.run for simpler integration if preferred for dev.
    try:
        import uvicorn
        uvicorn_args = [
            "app:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info",
        ]
        if reload_dev:
            uvicorn_args.append("--reload")
        else:
            # Number of workers; not typically used with --reload
            uvicorn_args.extend(["--workers", str(workers)])

        # uvicorn.run("app:app", host=host, port=port, workers=workers if not reload_dev else 1, reload=reload_dev, log_level="info")
        # Using subprocess for a slightly cleaner separation in some environments, but uvicorn.run() is also fine.
        # For this script, direct uvicorn.run is simpler:
        uvicorn.run(
            "app:app",  # module_name:app_instance_name
            host=host,
            port=port,
            workers=workers if not reload_dev else 1,
            reload=reload_dev,
            log_level="info"
        )

    except ImportError:
        print("❌ Uvicorn is not installed. Please install it: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()