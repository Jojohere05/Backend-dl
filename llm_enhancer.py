"""
LLM Enhancement Layer
Converts transformer output → User-friendly explanations matching frontend
"""

import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMEnhancer:
    """Use Gemini to generate explainability content for frontend"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.enabled = True
                print("✅ LLM enhancer enabled (Gemini)")
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {e}")
                self.enabled = False
        else:
            self.enabled = False
            print("⚠️  LLM enhancer disabled (no API key)")
    
    def enhance(self, pipeline_result):
        """
        Enhance pipeline output with explanations
        
        Args:
            pipeline_result: Output from CompletePipeline with predicted services
        
        Returns:
            Enhanced result matching frontend "Explainability" format
        """
        if not self.enabled:
            return self._generate_template_explanations(pipeline_result)
        
        try:
            return self._gemini_enhance(pipeline_result)
        except Exception as e:
            print(f"⚠️  LLM enhancement failed: {e}")
            return self._generate_template_explanations(pipeline_result)
    
    def _gemini_enhance(self, result):
        """Use Gemini to generate service explanations"""
        
        services = result['predicted_services']
        requirements = result.get('document_analysis', {}).get('simple_description', '')
        
        # Generate explanations for each service
        service_explanations = []
        
        for service_info in services:
            service = service_info['service']
            confidence = service_info['confidence']
            
            prompt = f"""You are an AWS solutions architect explaining service selection.


USER REQUIREMENT: {requirements}


SELECTED SERVICE: {service}
CONFIDENCE: {confidence:.0%}


Generate a brief explanation (2-3 sentences) explaining:
1. WHY this service was selected based on the requirements
2. What specific features make it suitable

Format: Direct explanation without headers or bullet points.
Tone: Professional but conversational."""

            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.3,
                        'max_output_tokens': 200
                    }
                )
                
                explanation = response.text.strip()
                
            except Exception as e:
                explanation = self._get_fallback_explanation(service, requirements)
            
            service_explanations.append({
                "service": service,
                "confidence": confidence,
                "category": service_info.get('category', 'other'),
                "description": service_info.get('description', ''),
                "explanation": explanation,
                "reasoning": self._extract_keywords(requirements, service)
            })
        
        return {
            **result,
            "explainability": {
                "service_explanations": service_explanations,
                "architecture_rationale": self._generate_architecture_rationale(result, requirements)
            }
        }
    
    def _generate_architecture_rationale(self, result, requirements):
        """Generate overall architecture explanation"""
        
        services = [s['service'] for s in result['predicted_services']]
        cost = result['metadata']['estimated_monthly_cost']
        
        prompt = f"""Explain the overall architecture design in 3-4 sentences.


REQUIREMENTS: {requirements}


SERVICES: {', '.join(services)}
ESTIMATED COST: ${cost}/month


Explain:
1. How these services work together
2. Why this architecture fits the requirements
3. Key architectural benefits

Format: Natural paragraph, no headers."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.3, 'max_output_tokens': 300}
            )
            return response.text.strip()
        except:
            return f"This architecture uses {len(services)} AWS services optimized for your requirements, with an estimated cost of ${cost}/month. The services are selected based on scalability, reliability, and cost-effectiveness."
    
    def _extract_keywords(self, requirements, service):
        """Extract relevant keywords from requirements for this service"""
        keywords = []
        
        req_lower = requirements.lower()
        
        # Service-specific keyword mapping
        keyword_map = {
            "S3": ["storage", "file", "image", "video", "photo", "media"],
            "Lambda": ["serverless", "function", "event", "trigger"],
            "DynamoDB": ["database", "NoSQL", "fast", "scalable"],
            "RDS": ["database", "SQL", "relational", "MySQL", "PostgreSQL"],
            "EC2": ["server", "compute", "VM", "instance"],
            "CloudFront": ["CDN", "distribution", "cache", "fast", "global"],
            "API_Gateway": ["API", "REST", "endpoint", "HTTP"],
            "Cognito": ["authentication", "user", "login", "signup"],
            "SNS": ["notification", "message", "alert", "push"],
            "SQS": ["queue", "async", "message"],
            "ECS": ["container", "docker", "microservices"],
            "VPC": ["network", "private", "security"],
            "IAM": ["permission", "access", "security", "role"]
        }
        
        for keyword in keyword_map.get(service, []):
            if keyword in req_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _get_fallback_explanation(self, service, requirements):
        """Fallback explanations when Gemini fails"""
        
        explanations = {
            "S3": "Selected for scalable object storage with high durability and low cost for storing files, images, and media assets.",
            "Lambda": "Chosen for serverless compute to run code without managing servers, automatically scaling based on demand.",
            "DynamoDB": "Selected for fast, scalable NoSQL database with single-digit millisecond response times and automatic scaling.",
            "RDS": "Chosen for managed relational database with automated backups, patches, and high availability.",
            "EC2": "Selected for flexible virtual servers with full control over computing resources and configuration.",
            "CloudFront": "Chosen for global content delivery network (CDN) to reduce latency and improve user experience worldwide.",
            "API_Gateway": "Selected for creating, deploying, and managing RESTful APIs with built-in security and throttling.",
            "Cognito": "Chosen for secure user authentication and authorization with support for social identity providers.",
            "SNS": "Selected for pub/sub messaging and push notifications to mobile and distributed systems.",
            "SQS": "Chosen for managed message queuing to decouple and scale microservices and serverless applications.",
            "ECS": "Selected for container orchestration to run Docker containers with automatic scaling and load balancing.",
            "VPC": "Chosen for isolated network environment with full control over IP addressing and security settings.",
            "IAM": "Selected for managing access permissions and policies across all AWS services securely."
        }
        
        return explanations.get(service, f"Selected for its capabilities relevant to your requirements.")
    
    def _generate_template_explanations(self, result):
        """Template-based explanations when Gemini is disabled"""
        
        service_explanations = []
        
        for service_info in result['predicted_services']:
            service = service_info['service']
            
            service_explanations.append({
                "service": service,
                "confidence": service_info['confidence'],
                "category": service_info.get('category', 'other'),
                "description": service_info.get('description', ''),
                "explanation": self._get_fallback_explanation(service, ""),
                "reasoning": []
            })
        
        return {
            **result,
            "explainability": {
                "service_explanations": service_explanations,
                "architecture_rationale": "This architecture is optimized based on your requirements and industry best practices."
            }
        }


# Test
if __name__ == "__main__":
    # Mock result matching your pipeline output
    mock_result = {
        "predicted_services": [
            {"service": "S3", "confidence": 0.95, "category": "storage", "description": "Object storage"},
            {"service": "Cognito", "confidence": 0.94, "category": "security", "description": "User authentication"},
            {"service": "Lambda", "confidence": 0.87, "category": "compute", "description": "Serverless functions"}
        ],
        "document_analysis": {
            "simple_description": "Build a web application for my med AI startup with photo-sharing and image storage features"
        },
        "metadata": {
            "estimated_monthly_cost": 234
        }
    }
    
    enhancer = LLMEnhancer()  # Will use GOOGLE_API_KEY from .env
    enhanced = enhancer.enhance(mock_result)
    
    print("="*80)
    print("SERVICE EXPLANATIONS (Frontend Format)")
    print("="*80)
    
    for svc in enhanced['explainability']['service_explanations']:
        print(f"\n{svc['service']} ({svc['category'].upper()})")
        print(f"Confidence: {svc['confidence']:.0%}")
        print(f"\nWe selected this because:")
        print(f"• {svc['explanation']}")
        if svc['reasoning']:
            print(f"\nMatched keywords: {', '.join(svc['reasoning'])}")
    
    print("\n" + "="*80)
    print("ARCHITECTURE RATIONALE")
    print("="*80)
    print(enhanced['explainability']['architecture_rationale'])
    
    print("\n✅ LLM Enhancer working!")
