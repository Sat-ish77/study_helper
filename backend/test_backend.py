from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client
import os
import requests

load_dotenv(Path('.env'))
sb = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

# Get token
resp = sb.auth.sign_in_with_password({
    'email': 'aadesh.wagle@gmail.com',
    'password': 'Satish@123'
})
token = resp.session.access_token
print('✅ Token obtained')

BASE = 'http://localhost:8000'
headers = {'Authorization': f'Bearer {token}'}

# Test 1 - Health
r = requests.get(f'{BASE}/health')
print('✅ Health:', r.json())

# Test 2 - List documents
r = requests.get(f'{BASE}/api/v1/rag/documents', headers=headers)
print('✅ Documents:', r.json())

# Test 3 - RAG ask
r = requests.post(
    f'{BASE}/api/v1/rag/ask',
    headers=headers,
    json={
        'question': 'tell me about my resume',
        'model': 'Llama 3.3 70B',
        'mode': 'medium',
        'web_fallback': True,
        'language': 'English',
        'history': []
    }
)
print('✅ RAG status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('Answer:', data['answer'][:200])
    print('Sources:', data['raw_sources'][:2])
else:
    print('Error:', r.text)

# Test 4 - Conversations list
r = requests.get(f'{BASE}/api/v1/conversations', headers=headers)
print('✅ Conversations:', r.json())

# Test 5 - Dashboard stats
r = requests.get(f'{BASE}/api/v1/dashboard/stats', headers=headers)
print('✅ Dashboard:', r.json())

# Test 6 - Flashcards due
r = requests.get(f'{BASE}/api/v1/flashcards?due_only=true', headers=headers)
print('✅ Flashcards due:', r.json())