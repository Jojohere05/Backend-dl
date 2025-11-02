"""
Inference Engine for 90% F1 Basic Transformer Model
Enhanced with hierarchical architecture diagram generation
"""

import torch
import pickle
import json
from sentence_transformers import SentenceTransformer
from model_transformer import TransformerServiceClassifier
import networkx as nx


class BasicArchitectureGenerator:
    """Use your 90% F1 model for predictions with architecture diagram generation"""
    
    def __init__(self, model_path='models/best_transformer_day2.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load checkpoint
        print("📦 Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get service names from label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
        self.service_names = self.mlb.classes_.tolist()
        
        # Initialize model
        print("🏗️  Initializing model...")
        self.model = TransformerServiceClassifier(
            input_dim=384,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            output_dim=len(self.service_names),
            dropout=0.3
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded (F1: {checkpoint.get('best_f1', 0.90):.4f})")
        
        # Load embedder
        print("📚 Loading Sentence-BERT...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load metadata
        with open('service.json', 'r') as f:
            self.services_data = json.load(f)
        
        print("✅ Inference engine ready!\n")
    
    def predict(self, text, threshold=0.5):
        """Predict services from text description"""
        embedding = self.embedder.encode([text], show_progress_bar=False)
        embedding_tensor = torch.FloatTensor(embedding).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(embedding_tensor)
            outputs = outputs.cpu().numpy()[0]
        
        predicted_indices = outputs > threshold
        predicted_services = [
            self.service_names[i] 
            for i in range(len(self.service_names)) 
            if predicted_indices[i]
        ]
        
        predicted_services = self._add_dependencies(predicted_services)
        result = self._build_output(text, predicted_services, outputs)
        
        return result
    
    def _add_dependencies(self, services):
        """Add required dependencies based on AWS service relationships"""
        
        dependency_map = {
            "EC2": ["VPC", "IAM", "Security_Groups"],
            "RDS": ["VPC", "IAM", "Security_Groups"],
            "Lambda": ["IAM", "CloudWatch"],
            "ECS": ["VPC", "IAM", "ECR", "CloudWatch"],
            "EKS": ["VPC", "IAM", "ECR", "CloudWatch"],
            "DynamoDB": ["IAM", "CloudWatch"],
            "API_Gateway": ["IAM", "CloudWatch"],
            "CloudFront": ["IAM", "S3"],
            "S3": ["IAM"],
            "Cognito": ["IAM"],
            "SNS": ["IAM", "CloudWatch"],
            "SQS": ["IAM", "CloudWatch"],
            "ElastiCache": ["VPC", "Security_Groups"],
            "Redshift": ["VPC", "IAM"],
            "Kinesis": ["IAM", "CloudWatch"],
            "Route53": ["IAM"],
            "ECR": ["IAM"],
            "CloudWatch": ["IAM"],
            "VPC": ["IAM"],
            "Security_Groups": ["VPC"]
        }
        
        all_services = set(services)
        for service in services:
            required = dependency_map.get(service, [])
            for dep in required:
                all_services.add(dep)
        
        return list(all_services)
    
    def _get_service_category(self, service):
        """Get category for dependency services not in model"""
        category_map = {
            "IAM": "security",
            "VPC": "networking",
            "CloudWatch": "monitoring",
            "Security_Groups": "security",
            "ECR": "container"
        }
        return category_map.get(service, "infrastructure")
    
    def _build_output(self, input_text, services, raw_outputs):
        """Build structured output with enhanced architecture graph"""
        service_details = []
        for service in services:
            # Handle dependency services not in model vocabulary
            if service not in self.service_names:
                category = self._get_service_category(service)
                service_details.append({
                    "service": service,
                    "confidence": 0.5,
                    "category": category,
                    "description": f"{service} (Infrastructure dependency)"
                })
                continue
            
            idx = self.service_names.index(service)
            
            category = None
            description = None
            for cat, svc_dict in self.services_data.items():
                if service in svc_dict:
                    category = cat
                    description = svc_dict[service].get('description', 'N/A')
                    break
            
            service_details.append({
                "service": service,
                "confidence": float(raw_outputs[idx]),
                "category": category,
                "description": description
            })
        
        service_details.sort(key=lambda x: x['confidence'], reverse=True)
        graph = self._build_hierarchical_graph(services, service_details)
        
        categories = {}
        for svc in service_details:
            cat = svc['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(svc['service'])
        
        return {
            "input_text": input_text,
            "predicted_services": service_details,
            "architecture_graph": graph,
            "total_services": len(services),
            "service_categories": categories
        }
    
    def _build_hierarchical_graph(self, services, service_details):
        """Build hierarchical architecture graph based on AWS best practices"""
        G = nx.DiGraph()
        
        # Create confidence map for layout positioning
        confidence_map = {svc['service']: svc['confidence'] for svc in service_details}
        
        # Add all predicted services as nodes with metadata
        for svc_detail in service_details:
            G.add_node(
                svc_detail['service'],
                category=svc_detail['category'],
                confidence=svc_detail['confidence'],
                description=svc_detail['description']
            )
        
        # Define architectural relationships
        relationships = self._define_service_relationships(services)
        
        # Add edges based on relationships
        for source, target, rel_type in relationships:
            if source in services and target in services:
                G.add_edge(source, target, type=rel_type)
        
        # Calculate hierarchical layout positions
        positions = self._calculate_hierarchical_positions(G, services, confidence_map)
        
        # Build output format
        nodes = []
        for node in G.nodes():
            node_data = G.nodes[node]
            nodes.append({
                "id": node,
                "name": node,
                "category": node_data.get('category', 'unknown'),
                "confidence": round(node_data.get('confidence', 0) * 100),
                "description": node_data.get('description', ''),
                "position": positions.get(node, {"x": 0, "y": 0, "layer": 0})
            })
        
        edges = []
        for source, target, data in G.edges(data=True):
            edges.append({
                "from": source,
                "to": target,
                "type": data.get('type', 'connects_to'),
                "label": self._get_edge_label(data.get('type', 'connects_to'))
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical",
            "layers": self._get_layer_info(positions)
        }
    
    def _define_service_relationships(self, services):
        """Define AWS service relationships based on typical architecture patterns"""
        relationships = []
        
        # Client → Frontend → Backend → Database flow
        if "CloudFront" in services:
            if "S3" in services:
                relationships.append(("CloudFront", "S3", "serves_from"))
            if "API_Gateway" in services:
                relationships.append(("CloudFront", "API_Gateway", "routes_to"))
        
        # API Gateway relationships
        if "API_Gateway" in services:
            if "Lambda" in services:
                relationships.append(("API_Gateway", "Lambda", "invokes"))
            if "EC2" in services:
                relationships.append(("API_Gateway", "EC2", "routes_to"))
            if "ECS" in services:
                relationships.append(("API_Gateway", "ECS", "routes_to"))
        
        # Compute → Database
        compute_services = ["Lambda", "EC2", "ECS", "EKS"]
        database_services = ["DynamoDB", "RDS", "ElastiCache", "Redshift"]
        
        for compute in compute_services:
            if compute in services:
                for db in database_services:
                    if db in services:
                        relationships.append((compute, db, "reads_writes"))
        
        # Compute → Storage
        for compute in compute_services:
            if compute in services and "S3" in services:
                relationships.append((compute, "S3", "stores_in"))
        
        # Authentication
        if "Cognito" in services:
            if "API_Gateway" in services:
                relationships.append(("Cognito", "API_Gateway", "authenticates"))
            for compute in compute_services:
                if compute in services:
                    relationships.append(("Cognito", compute, "authenticates"))
        
        # Messaging patterns
        if "SNS" in services:
            for compute in compute_services:
                if compute in services:
                    relationships.append((compute, "SNS", "publishes_to"))
        
        if "SQS" in services:
            for compute in compute_services:
                if compute in services:
                    relationships.append(("SQS", compute, "triggers"))
        
        # Monitoring (IAM and CloudWatch connect to everything)
        if "IAM" in services:
            for svc in services:
                if svc != "IAM":
                    relationships.append(("IAM", svc, "manages_access"))
        
        if "CloudWatch" in services:
            for svc in services:
                if svc != "CloudWatch" and svc not in ["IAM", "VPC", "Security_Groups"]:
                    relationships.append((svc, "CloudWatch", "logs_to"))
        
        # VPC and Security Groups
        vpc_services = ["EC2", "RDS", "ECS", "EKS", "ElastiCache", "Redshift"]
        if "VPC" in services:
            for svc in vpc_services:
                if svc in services:
                    relationships.append(("VPC", svc, "contains"))
        
        if "Security_Groups" in services:
            for svc in vpc_services:
                if svc in services:
                    relationships.append(("Security_Groups", svc, "protects"))
        
        return relationships
    
    def _calculate_hierarchical_positions(self, G, services, confidence_map):
        """Calculate hierarchical layer positions for architecture diagram"""
        
        # Define service layers (frontend → backend → data → support)
        layer_definition = {
            0: ["CloudFront", "Route_53"],
            1: ["S3", "API_Gateway"],
            2: ["Lambda", "EC2", "ECS", "EKS", "Cognito"],
            3: ["DynamoDB", "RDS", "ElastiCache", "Redshift", "SNS", "SQS", "Kinesis"],
            4: ["IAM", "CloudWatch", "VPC", "Security_Groups"]
        }
        
        positions = {}
        layer_counts = {}
        
        # Assign services to layers
        service_to_layer = {}
        for layer, layer_services in layer_definition.items():
            for svc in services:
                if svc in layer_services:
                    service_to_layer[svc] = layer
        
        # Services not in definition go to layer 2 (compute)
        for svc in services:
            if svc not in service_to_layer:
                service_to_layer[svc] = 2
        
        # Calculate positions
        for svc in services:
            layer = service_to_layer[svc]
            if layer not in layer_counts:
                layer_counts[layer] = 0
            
            # X position: spread horizontally within layer
            layer_service_count = sum(1 for s in services if service_to_layer[s] == layer)
            x_offset = (layer_counts[layer] - layer_service_count / 2) * 150
            
            # Y position: based on layer
            y_position = layer * 150
            
            # Add slight variation based on confidence
            confidence_offset = (confidence_map.get(svc, 0.5) - 0.5) * 20
            
            positions[svc] = {
                "x": x_offset,
                "y": y_position + confidence_offset,
                "layer": layer
            }
            
            layer_counts[layer] += 1
        
        return positions
    
    def _get_edge_label(self, edge_type):
        """Get human-readable label for edge type"""
        labels = {
            "serves_from": "Serves",
            "routes_to": "Routes",
            "invokes": "Invokes",
            "reads_writes": "R/W",
            "stores_in": "Stores",
            "authenticates": "Auth",
            "publishes_to": "Publishes",
            "triggers": "Triggers",
            "manages_access": "IAM",
            "logs_to": "Logs",
            "contains": "Contains",
            "protects": "Protects",
            "connects_to": "→"
        }
        return labels.get(edge_type, edge_type)
    
    def _get_layer_info(self, positions):
        """Get layer information for rendering"""
        layers = {}
        for node, pos in positions.items():
            layer = pos['layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        layer_names = {
            0: "CDN/DNS Layer",
            1: "Frontend/API Layer",
            2: "Compute/Auth Layer",
            3: "Data/Messaging Layer",
            4: "Infrastructure Layer"
        }
        
        return [
            {"layer": k, "name": layer_names.get(k, f"Layer {k}"), "services": v}
            for k, v in sorted(layers.items())
        ]
    
    def save_diagram_image(self, result, output_path='architecture_diagram.png'):
        """Save clean, hierarchical architecture diagram"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import matplotlib.patches as mpatches
        
        graph = result['architecture_graph']
        layers_info = graph['layers']
        
        # Create larger figure
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_xlim(-100, 2000)
        ax.set_ylim(-100, 1000)
        ax.axis('off')
        
        # Category colors (AWS-like)
        category_colors = {
            'compute': '#FF9900',      # AWS Orange
            'storage': '#3F8624',      # AWS Green
            'database': '#C925D1',     # AWS Purple
            'networking': '#527FFF',   # AWS Blue
            'security': '#DD344C',     # AWS Red
            'monitoring': '#759C3E',   # Light Green
            'messaging': '#E7157B',    # Pink
            'container': '#3334B9',    # Dark Blue
            'infrastructure': '#879196' # Gray
        }
        
        # Calculate vertical spacing
        layer_height = 180
        box_width = 180
        box_height = 80
        
        # Group services by layer
        node_positions = {}
        
        for layer_info in sorted(layers_info, key=lambda x: x['layer']):
            layer_num = layer_info['layer']
            services = layer_info['services']
            layer_name = layer_info['name']
            
            # Y position for this layer (top to bottom)
            y_base = 800 - (layer_num * layer_height)
            
            # Calculate horizontal spacing
            num_services = len(services)
            total_width = num_services * (box_width + 40)
            start_x = (1800 - total_width) / 2  # Center horizontally
            
            # Draw layer background
            layer_bg = mpatches.FancyBboxPatch(
                (50, y_base - 50),
                1850, layer_height - 30,
                boxstyle="round,pad=10",
                facecolor='#f8f9fa',
                edgecolor='#dee2e6',
                linewidth=2,
                alpha=0.3,
                zorder=0
            )
            ax.add_patch(layer_bg)
            
            # Layer label
            ax.text(70, y_base + 40, layer_name,
                    fontsize=14, fontweight='bold',
                    color='#495057',
                    verticalalignment='center')
            
            # Draw services in this layer
            for i, service_name in enumerate(services):
                x = start_x + (i * (box_width + 40))
                y = y_base
                
                node_positions[service_name] = (x + box_width/2, y)
                
                # Find service details
                service_data = next((s for s in result['predicted_services'] if s['service'] == service_name), None)
                if not service_data:
                    continue
                
                category = service_data['category']
                confidence = service_data['confidence']
                
                color = category_colors.get(category, '#6c757d')
                
                # Draw service box with shadow
                shadow = FancyBboxPatch(
                    (x + 3, y - 3), box_width, box_height,
                    boxstyle="round,pad=8",
                    facecolor='#00000020',
                    edgecolor='none',
                    zorder=1
                )
                ax.add_patch(shadow)
                
                box = FancyBboxPatch(
                    (x, y), box_width, box_height,
                    boxstyle="round,pad=8",
                    facecolor=color,
                    edgecolor='white',
                    linewidth=3,
                    alpha=0.9,
                    zorder=2
                )
                ax.add_patch(box)
                
                # Service name
                ax.text(x + box_width/2, y + box_height/2 + 5,
                        service_name.replace('_', ' '),
                        ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white',
                        zorder=3)
                
                # Confidence badge
                if confidence > 0.6:  # Only show for high-confidence services
                    badge_color = '#28a745' if confidence > 0.8 else '#ffc107'
                    confidence_badge = mpatches.Circle(
                        (x + box_width - 15, y + 15),
                        12,
                        facecolor=badge_color,
                        edgecolor='white',
                        linewidth=2,
                        zorder=4
                    )
                    ax.add_patch(confidence_badge)
                    ax.text(x + box_width - 15, y + 15,
                            f'{int(confidence * 100)}',
                            ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white',
                            zorder=5)
        
        # Draw connections (simplified - only show important ones)
        important_edge_types = ['invokes', 'reads_writes', 'authenticates', 'serves_from', 'routes_to']
        
        for edge in graph['edges']:
            if edge['type'] not in important_edge_types:
                continue  # Skip infrastructure connections (IAM, CloudWatch) for clarity
            
            from_pos = node_positions.get(edge['from'])
            to_pos = node_positions.get(edge['to'])
            
            if from_pos and to_pos:
                # Calculate control points for curved arrow
                mid_x = (from_pos[0] + to_pos[0]) / 2
                
                arrow = FancyArrowPatch(
                    from_pos, to_pos,
                    arrowstyle='->,head_width=8,head_length=10',
                    color='#6c757d',
                    linewidth=2.5,
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.2",
                    zorder=1
                )
                ax.add_patch(arrow)
                
                # Edge label
                ax.text(mid_x, (from_pos[1] + to_pos[1]) / 2 + 10,
                        edge['label'],
                        fontsize=9, color='#495057',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor='#dee2e6',
                                alpha=0.9),
                        zorder=2)
        
        # Title
        ax.text(950, 950, 'AWS Architecture Diagram',
                fontsize=22, fontweight='bold',
                ha='center', color='#212529')
        
        # Legend
        legend_x = 1600
        legend_y = 850
        
        ax.text(legend_x, legend_y + 30, 'Service Categories',
                fontsize=12, fontweight='bold', color='#212529')
        
        legend_items = sorted(category_colors.items())
        for i, (cat, color) in enumerate(legend_items):
            y = legend_y - (i * 30)
            
            legend_box = mpatches.Rectangle(
                (legend_x, y), 25, 20,
                facecolor=color,
                edgecolor='white',
                linewidth=2
            )
            ax.add_patch(legend_box)
            
            ax.text(legend_x + 35, y + 10, cat.title(),
                    fontsize=10, color='#495057',
                    verticalalignment='center')
        
        # Footer info
        ax.text(950, 30,
                f'Total Services: {result["total_services"]} | '
                f'Layers: {len(layers_info)} | '
                f'Connections: {len([e for e in graph["edges"] if e["type"] in important_edge_types])}',
                fontsize=10, color='#6c757d',
                ha='center')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Diagram saved to: {output_path}")
        plt.close()



if __name__ == "__main__":
    # Quick test
    generator = BasicArchitectureGenerator()
    result = generator.predict("Build a serverless API for mobile app with database")
    print(f"✅ Predicted {result['total_services']} services")
    print(f"✅ Graph has {len(result['architecture_graph']['nodes'])} nodes")
    print(f"✅ Graph has {len(result['architecture_graph']['edges'])} edges")
    
    # Save diagram
    generator.save_diagram_image(result, 'test_architecture.png')
