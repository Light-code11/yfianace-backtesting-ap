"""
Test script to diagnose Railway network connectivity to OpenAI
"""
import os
import sys
import socket
import requests
from dotenv import load_dotenv

load_dotenv()

def test_dns():
    """Test DNS resolution"""
    try:
        print("Testing DNS resolution for api.openai.com...")
        ip = socket.gethostbyname("api.openai.com")
        print(f"✅ DNS works: api.openai.com → {ip}")
        return True
    except Exception as e:
        print(f"❌ DNS failed: {e}")
        return False

def test_https():
    """Test HTTPS connection"""
    try:
        print("\nTesting HTTPS connection to OpenAI...")
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        print(f"✅ HTTPS works: Status {response.status_code}")
        return True
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ HTTPS failed: {e}")
        return False

def test_openai_auth():
    """Test OpenAI API with authentication"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ No OPENAI_API_KEY found")
        return False

    try:
        print(f"\nTesting OpenAI API with key: {api_key[:20]}...")
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )
        print(f"✅ OpenAI API works: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Railway Network Connectivity Test")
    print("=" * 60)

    dns_ok = test_dns()
    https_ok = test_https()
    api_ok = test_openai_auth()

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"DNS Resolution: {'✅ OK' if dns_ok else '❌ FAILED'}")
    print(f"HTTPS Connection: {'✅ OK' if https_ok else '❌ FAILED'}")
    print(f"OpenAI API: {'✅ OK' if api_ok else '❌ FAILED'}")

    if not https_ok:
        print("\n⚠️  Railway appears to be blocking outbound HTTPS connections")
        print("This is a Railway network configuration issue.")
        print("\nPossible solutions:")
        print("1. Contact Railway support about outbound connection blocking")
        print("2. Try deploying to a different Railway region")
        print("3. Use a different hosting platform (Render, Heroku, Fly.io)")
        print("4. Set up a proxy service")

if __name__ == "__main__":
    main()
