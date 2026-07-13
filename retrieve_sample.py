import os
import time
import json
import random
import requests
import pandas as pd
from urllib.parse import quote

# Configuration & Directories
IMAGE_DIR = "images"
DATA_DIR = "data"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_PATH = os.path.join(DATA_DIR, "met_api_cache.json")
METADATA_JSON_PATH = os.path.join(DATA_DIR, "met_drawings_sample.json")
METADATA_CSV_PATH = os.path.join(DATA_DIR, "met_drawings_sample.csv")

# Constants
DEPARTMENT_ID = 9  # Drawings and Prints
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
RATE_LIMIT_SLEEP = 0.025  # ~40 requests per second maximum

# Defined Cluster Hubs (Artists)
CLUSTERS = {
    "German Renaissance": ["Albrecht Dürer", "Lucas Cranach", "Hans Baldung"],
    "Dutch Golden Age": ["Rembrandt van Rijn", "Adriaen van Ostade"],
    "French 19th-Century": ["Honoré Daumier", "Edgar Degas", "Henri de Toulouse-Lautrec"]
}

# Local Caching System
def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

cache_db = load_cache()

# Robust GET Request with Cache, Retries and Rate-Limiting
def fetch_url(url):
    if url in cache_db:
        return cache_db[url]

    # Rate limiting sleep
    time.sleep(RATE_LIMIT_SLEEP)

    retries = 3
    backoff = 1.0
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                cache_db[url] = data
                save_cache(cache_db)
                return data
            elif response.status_code == 404:
                return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to fetch {url} after {retries} attempts: {e}")
                return None
        time.sleep(backoff)
        backoff *= 2.0
    return None

def download_image(url, filename):
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        return True # Already downloaded

    time.sleep(RATE_LIMIT_SLEEP)
    retries = 3
    backoff = 1.0
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to download image {url}: {e}")
                return False
        time.sleep(backoff)
        backoff *= 2.0
    return False

# Search and cluster objects
def collect_objects():
    target_sample_size = 45
    objects_per_cluster = target_sample_size // len(CLUSTERS)
    
    collected_objects = []

    for cluster_name, artists in CLUSTERS.items():
        print(f"\nProcessing Cluster: {cluster_name}")
        cluster_candidates = []

        for artist in artists:
            # Search for objects in Department 9 matching artist name
            query_url = f"{BASE_URL}/search?departmentId={DEPARTMENT_ID}&q={quote(artist)}"
            search_result = fetch_url(query_url)
            
            if not search_result or "objectIDs" not in search_result or not search_result["objectIDs"]:
                continue

            object_ids = search_result["objectIDs"]
            print(f"  Artist '{artist}': Found {len(object_ids)} search results. Filtering first 40...")

            # Filter for highlight/good matches first by checking a subset of candidate ids
            for obj_id in object_ids[:40]:
                obj_url = f"{BASE_URL}/objects/{obj_id}"
                obj_data = fetch_url(obj_url)
                
                if not obj_data:
                    continue
                
                # Check for criteria: Must have image, must be public domain
                has_image = obj_data.get("hasImages", False) or bool(obj_data.get("primaryImageSmall")) or bool(obj_data.get("primaryImage"))
                is_pd = obj_data.get("isPublicDomain", False)
                
                # Verify artist is actually the creator (avoiding books or prints *after* artist unless relevant)
                # We check if the display name matches or contains our artist string
                display_name = obj_data.get("artistDisplayName", "")
                artist_match = any(a.lower() in display_name.lower() for a in artists)

                if has_image and is_pd and artist_match:
                    cluster_candidates.append(obj_data)

        # Remove duplicates
        unique_candidates = {obj["objectID"]: obj for obj in cluster_candidates}.values()
        
        # Sort candidates so highlights and items with images are preferred
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda x: (x.get("isHighlight", False), bool(x.get("primaryImage"))),
            reverse=True
        )

        # Select a diverse sample of up to objects_per_cluster from this cluster
        selected = list(sorted_candidates)[:objects_per_cluster]
        print(f"  Selected {len(selected)} diverse, high-quality candidates for '{cluster_name}'.")
        
        for item in selected:
            item["cluster"] = cluster_name
            collected_objects.append(item)

    return collected_objects

def main():
    print("Starting MET Museum clustered sampling...")
    raw_objects = collect_objects()
    
    final_dataset = []
    
    print(f"\nDownloading images and extracting clean metadata for {len(raw_objects)} objects...")
    for idx, obj in enumerate(raw_objects, 1):
        obj_id = obj["objectID"]
        cluster_name = obj.get("cluster", "Unknown")
        
        # Decide which image URL to download (prefer primaryImageSmall for network optimization)
        img_url = obj.get("primaryImageSmall") or obj.get("primaryImage")
        if not img_url:
            continue
            
        # Extract file extension
        ext = img_url.split(".")[-1].split("?")[0]
        if ext.lower() not in ["jpg", "jpeg", "png", "gif"]:
            ext = "jpg"
            
        filename = f"met_{obj_id}.{ext}"
        
        # Download image
        success = download_image(img_url, filename)
        if not success:
            print(f"[{idx}/{len(raw_objects)}] Failed to download image for object {obj_id}.")
            continue

        # Extract specific metadata
        meta = {
            "object_id": obj_id,
            "title": obj.get("title", "Untitled"),
            "artist": obj.get("artistDisplayName", "Unknown Artist"),
            "date": obj.get("objectDate", "Date Unknown"),
            "place": obj.get("geoLocation", "Place Unknown") or obj.get("country", "Place Unknown"),
            "medium": obj.get("medium", "Unknown Medium"),
            "description": obj.get("medium", "No description available"),  # Medium acts as primary visual/physical description
            "culture": obj.get("culture", ""),
            "classification": obj.get("classification", ""),
            "is_highlight": obj.get("isHighlight", False),
            "cluster": cluster_name,
            "image_filename": filename,
            "image_url": img_url
        }
        
        final_dataset.append(meta)
        print(f"[{idx}/{len(raw_objects)}] Success: {meta['title']} ({meta['artist']})")

    # Save metadata as JSON
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON metadata to {METADATA_JSON_PATH}")

    # Save metadata as CSV
    df = pd.DataFrame(final_dataset)
    df.to_csv(METADATA_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved CSV metadata to {METADATA_CSV_PATH}")
    print(f"Total downloaded and logged objects: {len(final_dataset)}")

if __name__ == "__main__":
    main()
