"""
COMPLETE INTEGRATED PIPELINE
Document → Parser → Transformer (90% F1) → Cost Optimizer → LLM Enhancer → Output
All modules connected and production-ready
"""

import json
import os
from document_parser import RequirementsDocumentParser
from inference_basic import BasicArchitectureGenerator
from cost_optimizer import CostOptimizer
from llm_enhancer import LLMEnhancer


class CompletePipeline:
    """
    Full end-to-end pipeline integrating all components:
    1. Document Parser (Gemini)
    2. ML Transformer (90% F1)
    3. Cost Optimizer
    4. LLM Enhancer (Explainability)
    """
    
    def __init__(self, gemini_api_key=None):
        print("🚀 Initializing Complete Pipeline...")
        print("="*80)
        
        # Stage 1: Document Parser (Gemini)
        print("📄 Loading Document Parser...")
        self.parser = RequirementsDocumentParser(api_key=gemini_api_key, use_gemini=True)
        
        # Stage 2: ML Transformer (90% F1 Model)
        print("🤖 Loading ML Model...")
        self.generator = BasicArchitectureGenerator()
        
        # Stage 3: Cost Optimizer
        print("💰 Loading Cost Optimizer...")
        self.optimizer = CostOptimizer()
        
        # Stage 4: LLM Enhancer (Explainability)
        print("✨ Loading LLM Enhancer...")
        self.enhancer = LLMEnhancer(api_key=gemini_api_key)
        
        print("="*80)
        print("✅ Pipeline ready!\n")
    
    def process(self, document_text, budget_constraint="medium", use_explainability=True):
        """
        Complete processing pipeline
        
        Args:
            document_text: Raw document text (string) OR parsed result dict
            budget_constraint: "low", "medium", "high"
            use_explainability: Add LLM-generated explanations
        
        Returns:
            Complete architecture solution with all enhancements
        """
        
        # Check if input is already parsed
        if isinstance(document_text, dict) and 'simple_description' in document_text:
            parsed = document_text
            print("✓ Using pre-parsed document")
        else:
            print("\n🔄 STAGE 1: Parsing Document")
            print("-" * 80)
            parsed = self.parser.parse_document(document_text)
        
        print(f"✓ Description: {parsed['simple_description'][:80]}...")
        print(f"✓ Features: {parsed['technical_features']}")
        print(f"✓ Budget from doc: ${parsed.get('budget_monthly', 'N/A')}/month")
        print(f"✓ User count: {parsed['scale_indicators'].get('users', 'N/A')}")
        
        print("\n🔄 STAGE 2: Predicting Services (ML Model)")
        print("-" * 80)
        base_prediction = self.generator.predict(parsed['simple_description'], threshold=0.4)
        print(f"✓ Predicted {base_prediction['total_services']} services")
        print(f"✓ Services: {[s['service'] for s in base_prediction['predicted_services'][:8]]}")
        
        print("\n🔄 STAGE 3: Calculating Costs & Optimizing")
        print("-" * 80)
        services = [s['service'] for s in base_prediction['predicted_services']]
        
        # Determine traffic from scale
        scale = parsed['scale_indicators']
        traffic = "medium"
        if scale.get('users'):
            if scale['users'] < 10000:
                traffic = "low"
            elif scale['users'] > 50000:
                traffic = "high"
        
        print(f"✓ Traffic level: {traffic}")
        
        # Optimize for budget
        cost_result = self.optimizer.optimize_for_budget(
            services, 
            budget_constraint, 
            traffic
        )
        print(f"✓ {cost_result.get('message', 'Cost optimization complete')}")
        
        # Build result
        final_services = cost_result.get('optimized_services', services)
        
        result = {
            "document_analysis": parsed,
            "predicted_services": base_prediction['predicted_services'],
            "architecture_graph": base_prediction['architecture_graph'],
            "cost_optimization": cost_result,
            "final_services": final_services,
            "service_categories": base_prediction['service_categories'],
            "metadata": {
                "model_f1": 0.9052,
                "budget_tier": budget_constraint,
                "traffic_estimate": traffic,
                "estimated_monthly_cost": cost_result.get('optimized_cost', cost_result.get('current_cost', 0))
            }
        }
        
        # Stage 4: Add explainability
        if use_explainability:
            print("\n🔄 STAGE 4: Generating Explanations (LLM)")
            print("-" * 80)
            result = self.enhancer.enhance(result)
            print("✓ Service explanations generated")
        
        print("\n✅ Pipeline complete!")
        return result
    
    def process_from_file(self, file_path, budget_constraint="medium", use_explainability=True):
        """
        Process from uploaded file
        
        Args:
            file_path: Path to file (.txt, .pdf, .docx)
            budget_constraint: "low", "medium", "high"
            use_explainability: Add LLM-generated explanations
        
        Returns:
            Complete architecture solution
        """
        print(f"📄 Reading file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print("\n🔄 STAGE 1: Parsing Document from File")
        print("-" * 80)
        
        # Parse file - this returns the complete parsed result
        parsed = self.parser.parse_from_file(file_path)
        
        # Pass the parsed result to process (not just simple_description)
        return self.process(
            parsed,  # Pass entire parsed dict with all extracted info
            budget_constraint=budget_constraint,
            use_explainability=use_explainability
        )
    
    def save_result(self, result, output_file):
        """Save result to JSON"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved to: {output_file}")
    
    def print_summary(self, result):
        """Print human-readable summary"""
        print("\n" + "="*80)
        print("FINAL ARCHITECTURE SUMMARY")
        print("="*80)
        
        print(f"\n📊 Architecture Overview:")
        print(f"  Total Services: {len(result['final_services'])}")
        print(f"  Estimated Cost: ${result['metadata']['estimated_monthly_cost']}/month")
        print(f"  Budget Tier: {result['metadata']['budget_tier']}")
        print(f"  Traffic Level: {result['metadata']['traffic_estimate']}")
        
        print(f"\n🔧 Services by Category:")
        for category, services in result['service_categories'].items():
            print(f"  {category.upper()}: {', '.join(services)}")
        
        print(f"\n💰 Cost Optimization:")
        cost_opt = result['cost_optimization']
        print(f"  Status: {cost_opt['status']}")
        if 'savings' in cost_opt and cost_opt['savings'] > 0:
            print(f"  Original Cost: ${cost_opt.get('original_cost', 0)}/month")
            print(f"  Optimized Cost: ${cost_opt['optimized_cost']}/month")
            print(f"  Savings: ${cost_opt['savings']} ({cost_opt['savings_percentage']}%)")
            
            if cost_opt.get('changes_made'):
                print(f"\n  Changes Applied:")
                for change in cost_opt['changes_made'][:3]:
                    if change['type'] == 'removed':
                        print(f"    ❌ {change['service']}: {change['reason']}")
                        if change.get('alternative'):
                            print(f"       → Replaced with: {', '.join(change['alternative'])}")
        
        if 'explainability' in result:
            print(f"\n📝 Top Service Explanations:")
            for svc in result['explainability']['service_explanations'][:3]:
                print(f"\n  {svc['service']} ({svc['category']}) - Confidence: {svc['confidence']:.0%}")
                print(f"  {svc['explanation'][:120]}...")
        
        print("\n" + "="*80)


# Quick single test when run directly
if __name__ == "__main__":
    
    print("="*80)
    print("QUICK PIPELINE TEST")
    print("="*80)
    print()
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Quick test with direct text
    test_input = """
    Build an e-commerce platform with 100,000 daily users.
    Need product catalog, shopping cart, payment processing,
    user authentication, and order tracking.
    Budget: $2000/month
    """
    
    print(f"📝 Test Input: {test_input.strip()[:100]}...\n")
    
    result = pipeline.process(
        test_input,
        budget_constraint="medium",
        use_explainability=True
    )
    
    # Print summary
    pipeline.print_summary(result)
    
    # Save result
    os.makedirs('outputs', exist_ok=True)
    pipeline.save_result(result, 'outputs/quick_test_result.json')
    
    print("\n✅ Quick test complete!")
    print("💡 For comprehensive multi-domain testing, run: python test_all_domains.py")
