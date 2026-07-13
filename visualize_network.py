import os
import json
import networkx as nx
from pyvis.network import Network

# Path Configuration
DATA_DIR = "data"
IMAGE_DIR = "images"
METADATA_JSON_PATH = os.path.join(DATA_DIR, "met_drawings_sample.json")
OUTPUT_HTML_PATH = "art_semantic_network.html"

# Define border color mapping for clusters
CLUSTER_COLORS = {
    "German Renaissance": "#FFD700",  # Vivid Gold
    "Dutch Golden Age": "#E67E22",     # Amber Orange
    "French 19th-Century": "#E74C3C"   # Deep Crimson
}

def load_data():
    if not os.path.exists(METADATA_JSON_PATH):
        raise FileNotFoundError(f"Missing {METADATA_JSON_PATH}. Run retrieve_sample.py and gemini_enrichment.py first.")
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_network():
    dataset = load_data()
    
    # Initialize Pyvis Network
    # Use dark background to let golden, amber, and crimson borders pop!
    net = Network(height="750px", width="100%", bgcolor="#1C1C1E", font_color="white", notebook=False)
    
    # Precompute tag sets for each node to calculate edge intersections
    for item in dataset:
        tags = set()
        
        # Add Palette colors
        tags.update(c.lower().strip() for c in item.get("dominant_palette", []))
        # Add Visual Concepts
        tags.update(c.lower().strip() for c in item.get("visual_concepts", []))
        # Add Mood (single string)
        mood = item.get("mood")
        if mood and mood != "Unknown":
            tags.add(mood.lower().strip())
        # Add Research Themes
        tags.update(c.lower().strip() for c in item.get("research_themes", []))
        
        item["tag_set"] = tags

    # Build Nodes
    for item in dataset:
        obj_id = item["object_id"]
        title = item["title"]
        artist = item["artist"]
        date = item["date"]
        cluster = item["cluster"]
        image_file = item["image_filename"]
        
        # Construct the relative path to images directory for the html preview
        local_img_path = f"{IMAGE_DIR}/{image_file}"
        
        # Color coding borders by cluster (art movement)
        border_color = CLUSTER_COLORS.get(cluster, "#95A5A6")
        
        # Rich HTML Tooltip (Hover card)
        palette_html = "".join([f"<span style='background-color:#E0E0E0; color:#333; padding:2px 6px; margin:2px; border-radius:3px; font-size:11px; display:inline-block;'>{c}</span>" for c in item.get("dominant_palette", [])])
        concepts_html = ", ".join(item.get("visual_concepts", []))
        themes_html = ", ".join(item.get("research_themes", []))
        
        tooltip_html = f"""
        <div style="font-family: Arial, sans-serif; width: 300px; color: #333; background-color: #FFF; border-radius: 8px; padding: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border-top: 5px solid {border_color};">
            <h4 style="margin: 0 0 6px 0; color: #111; font-size: 14px;">{title}</h4>
            <p style="margin: 0 0 8px 0; font-size: 12px; color: #555;"><b>Artist:</b> {artist}<br><b>Date:</b> {date}<br><b>Movement:</b> {cluster}</p>
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="{local_img_path}" style="max-height: 120px; max-width: 100%; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"/>
            </div>
            <div style="font-size: 11px; margin-bottom: 6px;"><b>Palette:</b> {palette_html}</div>
            <div style="font-size: 11px; margin-bottom: 6px;"><b>Mood:</b> <span style="font-style: italic;">{item.get('mood', 'Unknown')}</span></div>
            <div style="font-size: 11px; margin-bottom: 4px;"><b>Concepts:</b> {concepts_html}</div>
            <div style="font-size: 11px;"><b>Themes:</b> {themes_html}</div>
        </div>
        """
        
        # Add node to network representation
        # shape="image" uses the artwork image itself
        net.add_node(
            n_id=str(obj_id),
            label=title[:25] + "..." if len(title) > 25 else title,
            shape="image",
            image=local_img_path,
            title=tooltip_html,
            borderWidth=4,
            borderWidthSelected=6,
            color={"border": border_color, "highlight": {"border": "#FFFFFF", "background": "#FFFFFF"}},
            size=30
        )

    # Build Edges (Pairs sharing >= 1 tag)
    edge_count = 0
    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            node_a = dataset[i]
            node_b = dataset[j]
            
            shared_tags = node_a["tag_set"] & node_b["tag_set"]
            weight = len(shared_tags)
            
            if weight >= 1:
                # Compile detail of overlapping tags for edge hover details
                overlap_list = ", ".join(list(shared_tags))
                edge_title = f"<b>Shared Tags ({weight}):</b><br>{overlap_list}"
                
                # Visual scaling: Thicker lines for higher overlap
                edge_width = weight * 1.5
                
                # Opacity based on strength (solid connections stand out more)
                opacity = min(0.3 + (weight * 0.15), 0.9)
                edge_color = f"rgba(255, 255, 255, {opacity})"
                
                net.add_edge(
                    source=str(node_a["object_id"]),
                    to=str(node_b["object_id"]),
                    value=weight,  # Scaling thickness
                    title=edge_title,
                    color={"color": edge_color, "highlight": "#FFF700"},
                    width=edge_width
                )
                edge_count += 1

    print(f"Constructed network with {len(dataset)} nodes and {edge_count} semantic edges.")

    # Configure physics engine to spread nodes nicely
    # Use Barneshut which is fast and handles clustered images smoothly
    net.set_options("""
    var options = {
      "edges": {
        "smooth": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.35,
          "springLength": 180,
          "springConstant": 0.04,
          "damping": 0.85,
          "avoidOverlap": 0.75
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # Save the output HTML file
    net.save_graph(OUTPUT_HTML_PATH)
    
    # Read the output HTML file and inject a custom DOM parser for tooltips
    # to force Vis.js to render them as actual formatted HTML instead of escaped text strings
    if os.path.exists(OUTPUT_HTML_PATH):
        with open(OUTPUT_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        tooltip_dom_script = """
                  // Convert string tooltips (titles) to DOM elements for rich HTML rendering in Vis.js
                  nodes.forEach(function(node) {
                    if (node.title) {
                      var div = document.createElement("div");
                      div.innerHTML = node.title;
                      node.title = div;
                    }
                  });
                  
                  // Same for edges
                  edges.forEach(function(edge) {
                    if (edge.title) {
                      var div = document.createElement("div");
                      div.innerHTML = edge.title;
                      edge.title = div;
                    }
                  });
        """
        
        marker = "data = {nodes: nodes, edges: edges};"
        if marker in html_content:
            html_content = html_content.replace(marker, tooltip_dom_script + "\n                  " + marker)
            
            # Inject physics optimization: turn off after stabilization
            marker2 = "network = new vis.Network(container, data, options);"
            physics_script = """
                  network.on("stabilizationIterationsDone", function () {
                      network.setOptions( { physics: false } );
                  });
            """
            if marker2 in html_content:
                html_content = html_content.replace(marker2, marker2 + "\n" + physics_script)

            with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
                f.write(html_content)
            print("Successfully injected custom DOM parser for HTML tooltips and physics optimization.")
            
            # Create a clean fragment for Quarto without html/head/body tags
            fragment = html_content.replace("<html>", "").replace("</html>", "")
            fragment = fragment.replace("<head>", "").replace("</head>", "")
            fragment = fragment.replace("<body>", "").replace("</body>", "")
            fragment = fragment.replace('<meta charset="utf-8">', "")
            
            with open("network_fragment.html", "w", encoding="utf-8") as f:
                f.write("```{=html}\n" + fragment + "\n```")
            print("Successfully created network_fragment.html for Quarto inclusion.")
        else:
            print("Warning: Tooltip marker not found in output HTML. Checking formatting.")

    print(f"Successfully generated interactive network graph at: {OUTPUT_HTML_PATH}")

if __name__ == "__main__":
    generate_network()
