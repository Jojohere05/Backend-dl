"""
Document Parser with File Upload Support
Handles: PDF, Word (.docx), TXT, and direct text input
"""

import re
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RequirementsDocumentParser:
    """Parse complex documents into structured requirements"""
    
    def __init__(self, api_key=None, use_gemini=True):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.use_gemini = use_gemini and self.api_key
        
        if self.use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.use_llm = True
                print("✅ LLM parser enabled (Google Gemini 2.0 Flash Experimental)")
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {e}")
                self.use_llm = False
        else:
            self.use_llm = False
            print("⚠️  Using fallback rule-based parser")
    
    def parse_from_file(self, file_path):
        """
        Parse document from uploaded file
        
        Args:
            file_path: Path to uploaded file (.txt, .pdf, .docx)
        
        Returns:
            Dict with extracted requirements
        """
        text = self._extract_text_from_file(file_path)
        
        if not text:
            raise ValueError(f"Could not extract text from {file_path}")
        
        return self.parse_document(text)
    
    def _extract_text_from_file(self, file_path):
        """Extract text from different file formats"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.txt':
                return self._extract_from_txt(file_path)
            elif file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        except Exception as e:
            print(f"⚠️  File extraction error: {e}")
            return None
    
    def _extract_from_txt(self, file_path):
        """Extract text from .txt file"""
        # Try multiple encodings to handle Windows files
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all fail, read as binary and decode with error handling
        with open(file_path, 'rb') as f:
            content = f.read()
            return content.decode('utf-8', errors='ignore')

    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF"""
        try:
            import PyPDF2
            
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            
            return '\n'.join(text)
        except ImportError:
            print("⚠️  PyPDF2 not installed. Install: pip install PyPDF2")
            return None
    
    def _extract_from_docx(self, file_path):
        """Extract text from Word document"""
        try:
            import docx
            
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            
            return '\n'.join(text)
        except ImportError:
            print("⚠️  python-docx not installed. Install: pip install python-docx")
            return None
    
    def parse_document(self, document_text):
        """Parse document text (from file or direct input)"""
        if self.use_llm:
            return self._parse_with_gemini(document_text)
        else:
            return self._parse_with_rules(document_text)
    
    def _parse_with_gemini(self, doc_text):
        """Use Google Gemini to parse document"""
        
        prompt = f"""You are an expert AWS solutions architect analyzing technical requirements.

DOCUMENT TO ANALYZE:
{doc_text}

TASK: Extract key information and return ONLY a valid JSON object (no markdown, no explanation, just pure JSON).

Required JSON structure:
{{
  "simple_description": "2-3 sentence summary describing the application in simple terms suitable for AWS architecture recommendation. Focus on: what the app does, scale, and key technical requirements.",
  "scale_indicators": {{
    "users": <number of daily active users or null>,
    "storage_tb": <storage in terabytes or null>,
    "concurrency": <peak concurrent users or null>
  }},
  "technical_features": [<array of feature strings from the list below>],
  "compute_preference": "<one of: serverless, traditional, gpu, or mixed>",
  "budget_monthly": <monthly budget in USD or null>
}}

FEATURE LIST to detect (include in technical_features array if found):
- "video_streaming" - if mentions video, streaming, media delivery
- "ai_ml" - if mentions AI, ML, recommendations, NLP, predictions
- "real_time" - if mentions real-time, live, websocket, instant updates
- "authentication" - if mentions auth, login, user management, SSO
- "file_upload" - if mentions file uploads, document storage
- "database" - if mentions database, data storage, persistence
- "messaging" - if mentions messaging, notifications, email, SMS, push
- "event_driven" - if mentions events, queues, async processing
- "analytics" - if mentions analytics, reporting, dashboards, metrics
- "mobile_backend" - if mentions mobile app, iOS, Android backend
- "static_hosting" - if mentions static site, HTML/CSS hosting

COMPUTE PREFERENCE rules:
- Use "gpu" if mentions GPU, ML training, heavy compute
- Use "serverless" if mentions Lambda, serverless, auto-scaling
- Use "traditional" if mentions EC2, servers, VMs
- Use "mixed" if mentions combination

Return ONLY the JSON object, no other text."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'top_p': 0.95,
                    'top_k': 40,
                }
            )
            
            content = response.text.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                print("✅ Gemini successfully parsed document")
                return parsed
            else:
                print("⚠️  No JSON found in Gemini response, using fallback")
                return self._parse_with_rules(doc_text)
                
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            return self._parse_with_rules(doc_text)
        except Exception as e:
            print(f"⚠️  Gemini parsing failed: {e}")
            return self._parse_with_rules(doc_text)
    
    def _parse_with_rules(self, doc_text):
        """Fallback: Rule-based parsing"""
        doc_lower = doc_text.lower()
        
        users = self._extract_number(doc_text, r'(\d+[,\d]*)\s*(?:daily\s*)?(?:active\s*)?users')
        storage = self._extract_number(doc_text, r'(\d+)\s*tb')
        concurrency = self._extract_number(doc_text, r'(\d+[,\d]*)\s*(?:peak\s*)?concurrency')
        budget = self._extract_number(doc_text, r'\$(\d+[,\d]*)')
        
        features = []
        feature_keywords = {
            'video_streaming': ['video', 'streaming', 'lecture', 'media'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'recommendation', 'nlp'],
            'real_time': ['real-time', 'realtime', 'live', 'websocket'],
            'authentication': ['auth', 'login', 'user management', 'signup'],
            'file_upload': ['upload', 'file storage', 'document'],
            'database': ['database', 'data storage', 'sql', 'nosql'],
            'messaging': ['messaging', 'notification', 'email', 'sms', 'push'],
            'event_driven': ['event', 'queue', 'asynchronous'],
            'analytics': ['analytics', 'reporting', 'dashboard', 'metrics'],
            'mobile_backend': ['mobile', 'app', 'ios', 'android'],
            'static_hosting': ['static', 'html', 'css', 'website']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(kw in doc_lower for kw in keywords):
                features.append(feature)
        
        compute_pref = "serverless"
        if 'gpu' in doc_lower or 'training' in doc_lower:
            compute_pref = "gpu"
        elif 'ec2' in doc_lower or 'server' in doc_lower:
            compute_pref = "traditional"
        elif 'serverless' in doc_lower or 'lambda' in doc_lower:
            compute_pref = "serverless"
        
        simple_desc = self._generate_simple_description(features, users, compute_pref)
        
        print("✅ Rule-based parser completed")
        
        return {
            "simple_description": simple_desc,
            "scale_indicators": {
                "users": users,
                "storage_tb": storage,
                "concurrency": concurrency
            },
            "technical_features": features,
            "compute_preference": compute_pref,
            "budget_monthly": budget
        }
    
    def _extract_number(self, text, pattern):
        """Extract number from text using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return int(num_str)
            except:
                return None
        return None
    
    def _generate_simple_description(self, features, users, compute_pref):
        """Generate simple text description from features"""
        parts = ["Build a"]
        
        if compute_pref == "serverless":
            parts.append("serverless")
        elif compute_pref == "gpu":
            parts.append("GPU-accelerated")
        
        if "video_streaming" in features:
            parts.append("video streaming platform")
        elif "mobile_backend" in features:
            parts.append("mobile backend")
        elif "static_hosting" in features:
            parts.append("static website")
        else:
            parts.append("web application")
        
        if users:
            if users < 10000:
                parts.append("for small-scale users")
            elif users < 50000:
                parts.append("for medium-scale users")
            else:
                parts.append("for large-scale users")
        
        feature_mentions = []
        if "authentication" in features:
            feature_mentions.append("user authentication")
        if "ai_ml" in features:
            feature_mentions.append("AI-powered recommendations")
        if "real_time" in features:
            feature_mentions.append("real-time interactions")
        if "file_upload" in features:
            feature_mentions.append("file storage")
        
        if feature_mentions:
            parts.append("with " + ", ".join(feature_mentions[:3]))
        
        return " ".join(parts) + "."
