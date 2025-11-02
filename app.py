"""
FastAPI Backend for Deep Cloud Architect
Connected to Lovable Frontend - Production Ready
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import os
import shutil
import traceback
from datetime import datetime

# Import your pipeline
from complete_pipeline import CompletePipeline

# Initialize FastAPI
app = FastAPI(
    title="Deep Cloud Architect API",
    description="AI-powered AWS architecture recommendation system",
    version="1.0.0"
)

# CORS Configuration - Allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Lovable local dev
        "http://localhost:8080",  # Lovable local dev alt
        "https://*.lovable.app",  # Lovable production
        "https://*.vercel.app",   # Vercel deployment
        "*"  # Allow all (remove in production, specify your domain)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline (loads models on startup)
print("="*80)
print("🚀 Initializing Deep Cloud Architect Backend...")
print("="*80)

try:
    pipeline = CompletePipeline()
    print("✅ Pipeline loaded successfully!")
except Exception as e:
    print(f"❌ Pipeline initialization failed: {e}")
    pipeline = None

# In-memory storage for saved architectures (replace with DB in production)
saved_architectures = []


# ============================================================================
# RESPONSE MODELS (Matching Frontend Expectations)
# ============================================================================

class ServiceDetail(BaseModel):
    """Individual service information"""
    service: str
    service_name: str  # Display name (e.g., "Amazon EC2")
    category: str
    confidence: float
    description: str


class ExplainabilityDetail(BaseModel):
    """Service explanation"""
    service: str
    confidence: float
    category: str
    explanation: str
    reasoning: List[str]


class CostBreakdown(BaseModel):
    """Cost per service"""
    service: str
    monthly_cost: float


class ArchitectureResponse(BaseModel):
    """Main response matching Lovable frontend structure"""
    success: bool
    services: List[ServiceDetail]
    architecture_diagram: Dict
    explainability: Dict
    cost_estimate: Dict
    metadata: Dict


# ============================================================================
# MAIN ENDPOINT: Generate Architecture
# ============================================================================

@app.post("/generate", response_model=ArchitectureResponse)
async def generate_architecture(
    file: UploadFile = File(..., description="Requirements document (.txt, .pdf, .docx)"),
    budget_range: str = Form(..., description="Budget: 'Low', 'Medium', or 'High'")
):
    """
    Main endpoint: Upload document + budget → Get AWS architecture
    
    Frontend sends:
    - file: User's requirements document
    - budget_range: Selected budget tier
    
    Backend returns:
    - services: List of predicted AWS services
    - architecture_diagram: Graph with nodes/edges
    - explainability: Why each service was selected
    - cost_estimate: Monthly cost breakdown
    """
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    print(f"\n{'='*80}")
    print(f"📥 New Request: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"📄 File: {file.filename}")
    print(f"💰 Budget: {budget_range}")
    
    # Validate budget
    budget_map = {
        "low": "low",
        "medium": "medium", 
        "high": "high",
        "Low": "low",
        "Medium": "medium",
        "High": "high"
    }
    budget_constraint = budget_map.get(budget_range, "medium")
    
    # Validate file type
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        print(f"✓ File saved to temp: {temp_file}")
        
        # Process with pipeline
        print("\n🔄 Processing with ML Pipeline...")
        result = pipeline.process_from_file(
            temp_file,
            budget_constraint=budget_constraint,
            use_explainability=True
        )
        
        print("\n✅ Pipeline processing complete!")
        
        # Transform to frontend format
        response = transform_to_frontend_format(result)
        
        print(f"\n📊 Generated architecture with {len(response['services'])} services")
        print(f"💵 Estimated cost: ${response['cost_estimate']['monthly_min']}-${response['cost_estimate']['monthly_max']}")
        print(f"{'='*80}\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error processing request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
            print(f"🗑️  Cleaned up temp file")


# ============================================================================
# HELPER: Transform Pipeline Output → Frontend Format
# ============================================================================

def transform_to_frontend_format(pipeline_result: Dict) -> Dict:
    """
    Transform pipeline output to match Lovable frontend expectations
    
    Pipeline output structure:
    {
        "predicted_services": [...],
        "architecture_graph": {"nodes": [...], "edges": [...]},
        "cost_optimization": {...},
        "explainability": {...}
    }
    
    Frontend expects:
    {
        "services": [{service, service_name, category, confidence, description}],
        "architecture_diagram": {nodes, edges},
        "explainability": {service_explanations: [...]},
        "cost_estimate": {monthly_min, monthly_max, breakdown}
    }
    """
    
    # 1. Transform Services
    services = []
    for svc in pipeline_result.get('predicted_services', []):
        services.append({
            "service": svc['service'],
            "service_name": f"Amazon {svc['service']}" if svc['service'] not in ['IAM', 'VPC'] else svc['service'],
            "category": svc.get('category', 'Other'),
            "confidence": round(svc['confidence'] * 100, 1),  # Convert to percentage
            "description": svc.get('description', 'AWS service')
        })
    
    # 2. Architecture Diagram (already in correct format)
    architecture_diagram = pipeline_result.get('architecture_graph', {
        "nodes": [],
        "edges": []
    })
    
    # 3. Explainability
    explainability = pipeline_result.get('explainability', {})
    
    # 4. Cost Estimate
    cost_data = pipeline_result.get('cost_optimization', {})
    cost_estimate = {
        "monthly_min": int(cost_data.get('optimized_cost', cost_data.get('current_cost', 100)) * 0.8),
        "monthly_max": int(cost_data.get('optimized_cost', cost_data.get('current_cost', 100)) * 1.2),
        "currency": "USD",
        "breakdown": cost_data.get('breakdown', []),
        "status": cost_data.get('status', 'estimated')
    }
    
    # 5. Metadata
    metadata = {
        "model_f1_score": 0.9052,
        "budget_tier": pipeline_result.get('metadata', {}).get('budget_tier', 'medium'),
        "total_services": len(services),
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "success": True,
        "services": services,
        "architecture_diagram": architecture_diagram,
        "explainability": explainability,
        "cost_estimate": cost_estimate,
        "metadata": metadata
    }


# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@app.post("/save-architecture")
async def save_architecture(data: Dict):
    """Save generated architecture to user library"""
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
            "message": "Architecture saved successfully",
            "id": architecture_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """Get saved architectures (History tab)"""
    return {
        "success": True,
        "architectures": saved_architectures,
        "count": len(saved_architectures)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
        "docs": "/docs"
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print("🚀 Deep Cloud Architect API Started")
    print("="*80)
    print(f"📡 Listening on: http://0.0.0.0:8000")
    print(f"📚 API Docs: http://0.0.0.0:8000/docs")
    print(f"🔧 Pipeline Status: {'✅ Loaded' if pipeline else '❌ Not Loaded'}")
    print("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n🛑 Shutting down Deep Cloud Architect API...")


# ============================================================================
# RUN SERVER (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
