import os
import sys
import json
import base64
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

IMAGE_DIR = "images"
DATA_DIR = "data"
METADATA_JSON_PATH = os.path.join(DATA_DIR, "met_drawings_sample.json")
METADATA_CSV_PATH = os.path.join(DATA_DIR, "met_drawings_sample.csv")

# Initialize OpenAI client with Litellm configuration
key = os.getenv("GEMINI_API_KEY")
if not key:
    print("CRITICAL: GEMINI_API_KEY is not defined. Please add it to your .env file.")
    sys.exit(1)

client = OpenAI(
    api_key=key,
    base_url="https://litellm.dreamlab.ucsb.edu/v1"
)

# Helper to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_gemini(image_path, title, artist):
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}")
        return None

    base64_image = encode_image(image_path)
    
    # Prompt instructing Gemini to return exact structured JSON format
    prompt = f"""You are an expert art historian and visual analyst.
Analyze the following drawing/print from the MET Museum:
Title: {title}
Artist: {artist}

You must return a valid JSON object containing exactly the following keys:
1. "dominant_palette": An array of exactly 3 distinct visual colors (e.g., ["warm gold", "deep indigo", "creamy parchment"]).
2. "visual_concepts": An array of primary subjects/objects/motifs visible in the work (e.g., ["allegorical figure", "hourglass", "landscape", "dog"]).
3. "mood": A single string representing the overall emotional tone or vibe (e.g., "melancholic", "serene", "triumphant", "chaotic").
4. "research_themes": An array of latent historical/scholarly themes (e.g., ["mortality", "scientific inquiry", "humanism", "domesticity"]).

Do not include any Markdown, backticks (e.g., ```json), preambles, or explanations. Only return a raw JSON parseable object.
"""

    retries = 3
    backoff = 2.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gemini-3.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting block wrapper if model returned any
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            parsed_data = json.loads(raw_text)
            
            # Basic structural verification
            required_keys = ["dominant_palette", "visual_concepts", "mood", "research_themes"]
            if all(key in parsed_data for key in required_keys):
                return parsed_data
            else:
                print(f"Warning: Response missing keys for {title}. Retrying...")
                
        except json.JSONDecodeError as jde:
            print(f"JSON decode error: {jde}. Raw text: {raw_text[:100]}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
        time.sleep(backoff)
        backoff *= 2.0
        
    return None

def main():
    print("Starting Gemini Multimodal metadata enrichment...")
    
    if not os.path.exists(METADATA_JSON_PATH):
        print(f"Error: {METADATA_JSON_PATH} not found. Please run retrieve_sample.py first.")
        sys.exit(1)
        
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    enriched_dataset = []
    
    print(f"\nProcessing {len(dataset)} items through Gemini...")
    for idx, obj in enumerate(dataset, 1):
        title = obj.get("title", "Untitled")
        artist = obj.get("artist", "Unknown")
        image_filename = obj.get("image_filename")
        
        # Check if already enriched to skip API call and be idempotent
        if "mood" in obj and obj["mood"] != "Unknown" and len(obj.get("dominant_palette", [])) == 3:
            print(f"[{idx}/{len(dataset)}] Skipping (already enriched): {title}")
            enriched_dataset.append(obj)
            continue
            
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        print(f"[{idx}/{len(dataset)}] Analyzing: {title} by {artist} ({image_filename})...")
        
        analysis = analyze_image_with_gemini(image_path, title, artist)
        
        if analysis:
            # Merge returned attributes into the object metadata
            obj["dominant_palette"] = analysis["dominant_palette"]
            obj["visual_concepts"] = analysis["visual_concepts"]
            obj["mood"] = analysis["mood"]
            obj["research_themes"] = analysis["research_themes"]
            print(f"  -> Palette: {analysis['dominant_palette']}")
            print(f"  -> Mood: {analysis['mood']}")
        else:
            # Use placeholders if analysis failed
            obj["dominant_palette"] = ["Unknown", "Unknown", "Unknown"]
            obj["visual_concepts"] = []
            obj["mood"] = "Unknown"
            obj["research_themes"] = []
            print(f"  -> [Failed to analyze, used placeholders]")
            
        enriched_dataset.append(obj)
        
        # Save intermediate progress to JSON continuously to prevent data loss on timeouts
        with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(enriched_dataset + dataset[idx:], f, indent=2, ensure_ascii=False)
            
        # Sleep slightly to remain friendly to the proxy rates
        time.sleep(0.5)

    # Save final enriched dataset as JSON
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved enriched JSON metadata to {METADATA_JSON_PATH}")

    # Convert complex lists to string format for neat CSV representation
    csv_dataset = []
    for item in enriched_dataset:
        csv_item = item.copy()
        # Convert lists to comma-separated strings for tidy spreadsheet usage
        csv_item["dominant_palette"] = ", ".join(csv_item["dominant_palette"])
        csv_item["visual_concepts"] = ", ".join(csv_item["visual_concepts"])
        csv_item["research_themes"] = ", ".join(csv_item["research_themes"])
        csv_dataset.append(csv_item)

    # Save enriched dataset as CSV
    df = pd.DataFrame(csv_dataset)
    df.to_csv(METADATA_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved enriched CSV metadata to {METADATA_CSV_PATH}")
    print("Enrichment process completed successfully!")

if __name__ == "__main__":
    main()
