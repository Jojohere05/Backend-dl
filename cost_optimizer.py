"""
Enhanced Cost Optimizer using AWS Cost Calculator
"""

import json
from aws_cost_calculator import AWSCostCalculator  # Import calculator


class CostOptimizer:
    """Optimize AWS architecture for budget constraints"""
    
    def __init__(self):  # FIXED: __init__
        # Use AWS cost calculator for accurate pricing
        self.calculator = AWSCostCalculator()
        
        # Service alternatives for cost optimization
        self.alternatives = {
            "EC2": {
                "cheaper": ["Lambda", "ECS"],
                "reason": "Serverless reduces idle costs by 60-80%",
                "tradeoff": "Limited to 15min execution time"
            },
            "RDS": {
                "cheaper": ["DynamoDB"],
                "reason": "NoSQL can be 40-60% cheaper for simple use cases",
                "tradeoff": "No complex joins, different data model"
            },
            "ECS": {
                "cheaper": ["Lambda"],
                "reason": "Serverless avoids container orchestration costs",
                "tradeoff": "Less control over environment"
            }
        }
    
    def estimate_cost(self, services, traffic="medium"):
        """
        Estimate monthly cost using AWS Cost Calculator
        
        Args:
            services: List of service names
            traffic: "low", "medium", "high"
        
        Returns:
            Estimated monthly cost in USD
        """
        # Use AWS calculator for accurate costs
        result = self.calculator.estimate_monthly_cost(services, usage_profile=traffic)
        return result['total_monthly']
    
    def optimize_for_budget(self, predicted_services, budget_constraint, traffic_estimate="medium"):
        """
        Optimize architecture to fit budget
        
        Args:
            predicted_services: List of predicted service names
            budget_constraint: "low" (<$100), "medium" ($100-500), "high" (>$500)
            traffic_estimate: "low", "medium", "high"
        
        Returns:
            Dict with optimization results
        """
        budget_limits = {
            "low": 100,
            "medium": 500,
            "high": 2000
        }
        
        target_budget = budget_limits.get(budget_constraint, 500)
        current_cost = self.estimate_cost(predicted_services, traffic_estimate)
        
        if current_cost <= target_budget:
            # Get detailed breakdown
            breakdown = self.calculator.estimate_monthly_cost(predicted_services, traffic_estimate)
            
            return {
                "status": "within_budget",
                "current_cost": current_cost,
                "target_budget": target_budget,
                "services": predicted_services,
                "breakdown": breakdown['breakdown'],
                "changes_made": [],
                "message": f"✅ Architecture fits within ${target_budget}/month budget"
            }
        
        # Need optimization
        print(f"💰 Current cost ${current_cost} exceeds budget ${target_budget}")
        print(f"🔄 Applying cost optimizations...")
        
        optimized_services = self._apply_optimizations(
            predicted_services.copy(),
            current_cost,
            target_budget,
            traffic_estimate
        )
        
        new_cost = self.estimate_cost(optimized_services, traffic_estimate)
        changes = self._get_changes(predicted_services, optimized_services)
        
        # Get detailed breakdown
        breakdown = self.calculator.estimate_monthly_cost(optimized_services, traffic_estimate)
        
        return {
            "status": "optimized" if new_cost <= target_budget else "partial_optimization",
            "original_cost": current_cost,
            "optimized_cost": new_cost,
            "target_budget": target_budget,
            "savings": round(current_cost - new_cost, 2),
            "savings_percentage": round(((current_cost - new_cost) / current_cost) * 100, 1),
            "original_services": predicted_services,
            "optimized_services": optimized_services,
            "breakdown": breakdown['breakdown'],
            "changes_made": changes,
            "message": self._get_message(new_cost, target_budget, current_cost)
        }
    
    def _apply_optimizations(self, services, current_cost, target_budget, traffic):
        """Apply cost reduction strategies"""
        
        # Strategy 1: Replace expensive compute with serverless
        if current_cost > target_budget * 1.3:
            if "EC2" in services:
                services.remove("EC2")
                if "Lambda" not in services:
                    services.append("Lambda")
                if "API_Gateway" not in services:
                    services.append("API_Gateway")
                print("  ✓ Replaced EC2 with Lambda (serverless)")
        
        # Strategy 2: Use cheaper database
        if self.estimate_cost(services, traffic) > target_budget:
            if "RDS" in services:
                services.remove("RDS")
                if "DynamoDB" not in services:
                    services.append("DynamoDB")
                print("  ✓ Replaced RDS with DynamoDB")
        
        # Strategy 3: Remove optional expensive services
        optional_expensive = ["ECS", "EBS"]
        for svc in optional_expensive:
            if svc in services and self.estimate_cost(services, traffic) > target_budget:
                services.remove(svc)
                print(f"  ✓ Removed optional service: {svc}")
        
        # Strategy 4: Remove CDN if still over budget
        if self.estimate_cost(services, traffic) > target_budget:
            if "CloudFront" in services:
                services.remove("CloudFront")
                print("  ✓ Removed CloudFront (CDN)")
        
        # Ensure IAM is always present
        if "IAM" not in services:
            services.append("IAM")
        
        return services
    
    def _get_changes(self, original, optimized):
        """Get list of changes made"""
        changes = []
        
        removed = set(original) - set(optimized)
        added = set(optimized) - set(original)
        
        for svc in removed:
            alt_info = self.alternatives.get(svc, {})
            changes.append({
                "type": "removed",
                "service": svc,
                "reason": alt_info.get("reason", "Cost reduction"),
                "alternative": alt_info.get("cheaper", []),
                "tradeoff": alt_info.get("tradeoff", "None")
            })
        
        for svc in added:
            if svc not in original:
                changes.append({
                    "type": "added",
                    "service": svc,
                    "reason": "Cost-effective alternative"
                })
        
        return changes
    
    def _get_message(self, new_cost, target, original):
        """Generate status message"""
        if new_cost <= target:
            savings = original - new_cost
            return f"✅ Optimized to ${new_cost}/month (saved ${round(savings, 2)}, {round((savings/original)*100)}%)"
        else:
            return f"⚠️  Partially optimized to ${new_cost}/month (still ${round(new_cost - target, 2)} over budget)"


# Test
if __name__ == "__main__":  # FIXED
    optimizer = CostOptimizer()
    
    print("="*80)
    print("TESTING ENHANCED COST OPTIMIZER")
    print("="*80)
    
    expensive_services = ["EC2", "RDS", "ECS", "S3", "CloudFront", "VPC", "IAM"]
    
    print("\n📊 TEST 1: Expensive Architecture")
    print(f"Services: {expensive_services}")
    
    result = optimizer.optimize_for_budget(expensive_services, "low", "medium")
    
    print(f"\nStatus: {result['status']}")
    print(f"Original cost: ${result['original_cost']}/month")
    print(f"Optimized cost: ${result['optimized_cost']}/month")
    print(f"Savings: ${result['savings']} ({result['savings_percentage']}%)")
    print(f"\nOptimized services: {result['optimized_services']}")
    
    print("\n💰 Cost Breakdown:")
    for item in result['breakdown'][:5]:
        print(f"  {item['service']:15s} ${item['monthly_cost']:>8.2f}/month")
