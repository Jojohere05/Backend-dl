"""
FastAPI Backend for Deep Cloud Architect
Connected to Lovable Frontend - Railway Production Ready
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import os
import shutil
import traceback
from datetime import datetime
from PyPDF2 import PdfReader

# Import your pipeline
from complete_pipeline import CompletePipeline

# Initialize FastAPI
app = FastAPI(
    title="Deep Cloud Architect API",
    description="AI-powered AWS architecture recommendation system",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Lovable domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("="*80)
    print("🚀 Initializing Deep Cloud Architect Backend...")
    print("="*80)
    try:
        pipeline = CompletePipeline()
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        traceback.print_exc()

# In-memory storage (replace with DB in production)
saved_architectures = []

# Output directory for generated files
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.pdf':
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif file_ext in ['.docx', '.doc']:
        # For DOCX support, install: pip install python-docx
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except ImportError:
            raise HTTPException(400, "DOCX support not available. Install python-docx")
    
    else:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")


def transform_to_frontend_format(pipeline_result: Dict) -> Dict:
    """Transform pipeline output to match Lovable frontend"""
    
    # 1. Services with full details
    services = []
    for svc in pipeline_result.get('predicted_services', []):
        services.append({
            "service": svc['service'],
            "service_name": get_service_display_name(svc['service']),
            "category": svc.get('category', 'Other'),
            "confidence": round(svc['confidence'] * 100, 1),
            "description": get_service_description(svc['service'])
        })
    
    # 2. Architecture diagram
    architecture_diagram = pipeline_result.get('architecture_graph', {
        "nodes": [],
        "edges": [],
        "layers": []
    })
    
    # 3. Explainability
    explainability_raw = pipeline_result.get('explainability', {})
    explainability = {
        "architecture_rationale": explainability_raw.get('architecture_rationale', 'Architecture optimized for your requirements.'),
        "service_explanations": explainability_raw.get('service_explanations', [])
    }
    
    # 4. Cost estimate
    cost_data = pipeline_result.get('cost_optimization', {})
    monthly_cost = pipeline_result.get('metadata', {}).get('estimated_monthly_cost', 100)
    
    cost_estimate = {
        "monthly_min": int(monthly_cost * 0.85),
        "monthly_max": int(monthly_cost * 1.15),
        "currency": "USD",
        "status": cost_data.get('status', 'estimated'),
        "breakdown": []
    }
    
    # 5. Metadata
    metadata = {
        "model_type": "Deep Learning Transformer",
        "model_f1_score": 0.9115,
        "budget_tier": pipeline_result.get('metadata', {}).get('budget_tier', 'medium'),
        "total_services": len(services),
        "timestamp": datetime.now().isoformat(),
        "traffic_estimate": pipeline_result.get('metadata', {}).get('traffic_estimate', 'medium')
    }
    
    return {
        "success": True,
        "services": services,
        "architecture_diagram": architecture_diagram,
        "explainability": explainability,
        "cost_estimate": cost_estimate,
        "metadata": metadata
    }


def get_service_display_name(service: str) -> str:
    """Get user-friendly service name"""
    display_names = {
        'EC2': 'Amazon EC2',
        'S3': 'Amazon S3',
        'RDS': 'Amazon RDS',
        'Lambda': 'AWS Lambda',
        'DynamoDB': 'Amazon DynamoDB',
        'CloudFront': 'Amazon CloudFront',
        'IAM': 'AWS IAM',
        'VPC': 'Amazon VPC',
        'API_Gateway': 'Amazon API Gateway',
        'CloudWatch': 'Amazon CloudWatch',
        'SNS': 'Amazon SNS',
        'SQS': 'Amazon SQS',
        'Cognito': 'Amazon Cognito',
        'ECS': 'Amazon ECS',
        'EKS': 'Amazon EKS'
    }
    return display_names.get(service, f'Amazon {service}')


def get_service_description(service: str) -> str:
    """Get service description"""
    descriptions = {
        'EC2': 'Scalable virtual servers in the cloud',
        'S3': 'Object storage with high durability',
        'RDS': 'Managed relational database service',
        'Lambda': 'Run code without managing servers',
        'DynamoDB': 'Fast, flexible NoSQL database',
        'CloudFront': 'Global content delivery network',
        'IAM': 'Identity and access management',
        'VPC': 'Isolated cloud network',
        'API_Gateway': 'Create and manage APIs',
        'CloudWatch': 'Monitoring and observability'
    }
    return descriptions.get(service, 'AWS cloud service')


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.post("/generate")
async def generate_architecture(
    file: UploadFile = File(...),
    budget_range: str = Form(...)
):
    """Generate AWS architecture from uploaded document"""
    
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    
    print(f"\n{'='*80}")
    print(f"📥 New Request: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"📄 File: {file.filename}")
    print(f"💰 Budget: {budget_range}")
    
    # Budget mapping
    budget_map = {
        "low": "low", "Low": "low",
        "medium": "medium", "Medium": "medium",
        "high": "high", "High": "high"
    }
    budget_constraint = budget_map.get(budget_range, "medium")
    
    # Validate file
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")
    
    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    temp_file = None
    
    try:
        # Save uploaded file
        temp_file = os.path.join(session_dir, file.filename)
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ File saved: {temp_file}")
        
        # Extract text
        print("📖 Extracting text...")
        document_text = extract_text_from_file(temp_file, file.filename)
        print(f"✓ Extracted {len(document_text)} characters")
        
        if len(document_text) < 50:
            raise HTTPException(400, "Document too short (min 50 characters)")
        
        # Run pipeline (FIXED: uses .process() with text, not file path)
        print("\n🔄 Processing with ML Pipeline...")
        result = pipeline.process(
            document_text,  # Pass text, not file path
            budget_constraint=budget_constraint,
            use_explainability=True  # Enable for Lovable frontend
        )
        
        print("\n✅ Pipeline processing complete!")
        
        # Save diagram
        try:
            from inference_basic import BasicArchitectureGenerator
            gen = BasicArchitectureGenerator()
            diagram_data = {
                'architecture_graph': result['architecture_graph'],
                'predicted_services': result['predicted_services'],
                'total_services': len(result['final_services'])
            }
            diagram_path = os.path.join(session_dir, "diagram.png")
            gen.save_diagram_image(diagram_data, diagram_path)
            print(f"✓ Diagram saved: {diagram_path}")
        except Exception as e:
            print(f"⚠️  Diagram generation failed: {e}")
        
        # Transform to frontend format
        response = transform_to_frontend_format(result)
        response['session_id'] = session_id
        response['diagram_url'] = f"/api/diagram/{session_id}"
        
        print(f"\n📊 Generated architecture:")
        print(f"   Services: {len(response['services'])}")
        print(f"   Cost: ${response['cost_estimate']['monthly_min']}-${response['cost_estimate']['monthly_max']}")
        print(f"{'='*80}\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temp file only (keep session dir for diagram)
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@app.get("/api/diagram/{session_id}")
async def get_diagram(session_id: str):
    """Get generated diagram"""
    diagram_path = os.path.join(OUTPUT_DIR, session_id, "diagram.png")
    
    if not os.path.exists(diagram_path):
        raise HTTPException(404, "Diagram not found")
    
    return FileResponse(diagram_path, media_type="image/png")


@app.post("/save-architecture")
async def save_architecture(data: Dict):
    """Save architecture to history"""
    try:
        architecture_id = len(saved_architectures) + 1
        saved_data = {
            "id": architecture_id,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        saved_architectures.append(saved_data)
        
        return {
            "success": True,
            "message": "Architecture saved",
            "id": architecture_id
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
async def get_history():
    """Get saved architectures"""
    return {
        "success": True,
        "architectures": saved_architectures,
        "count": len(saved_architectures)
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Deep Cloud Architect API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "pipeline": "ready" if pipeline else "not_ready"
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Disable in production
    )
