"""
Test script for FastAPI backend
Tests file upload and architecture generation
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Health check passed!")


def test_generate_architecture():
    """Test architecture generation with file upload"""
    print("\n" + "="*80)
    print("TEST 2: Generate Architecture")
    print("="*80)
    
    # Create test document
    test_content = """
    Project Name: EduVision — Online Coaching for Entrance Exams
    
    Overview: EduVision is an EdTech startup providing live interactive 
    classes and recorded lectures for students preparing for entrance exams.
    
    Expected Scale:
    • Daily active users: ~50,000
    • Peak concurrency: ~3,000 users
    • Storage needs: ~20 TB (lecture videos and transcripts)
    • Monthly budget: ~$8,000
    
    Technical Requirements:
    • Video streaming for lectures
    • User authentication and profiles
    • AI-driven personalized course recommendations
    • Real-time chat during live sessions
    • Secure student data management
    """
    
    # Save to temp file
    test_file_path = "test_requirements.txt"
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    # Prepare request
    files = {'file': open(test_file_path, 'rb')}
    data = {'budget_range': 'Medium'}
    
    # Send request
    print(f"\n📤 Sending request to {BASE_URL}/generate")
    response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
    
    print(f"\n📥 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Architecture Generated!")
        print(f"\nServices ({len(result['services'])} total):")
        for svc in result['services'][:5]:
            print(f"  • {svc['service_name']}: {svc['confidence']}% confidence")
        
        print(f"\nCost Estimate:")
        print(f"  ${result['cost_estimate']['monthly_min']} - ${result['cost_estimate']['monthly_max']}/month")
        
        print(f"\nArchitecture Diagram:")
        print(f"  Nodes: {len(result['architecture_diagram']['nodes'])}")
        print(f"  Edges: {len(result['architecture_diagram']['edges'])}")
        
        print("\n✅ Test passed!")
    else:
        print(f"\n❌ Test failed!")
        print(f"Error: {response.text}")
    
    # Cleanup
    import os
    os.remove(test_file_path)


if __name__ == "__main__":
    print("="*80)
    print("DEEP CLOUD ARCHITECT - BACKEND TESTING")
    print("="*80)
    
    try:
        test_health()
        test_generate_architecture()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
