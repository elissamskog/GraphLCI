import requests

def search_hf(query):
    url = f"https://huggingface.co/api/datasets?search={query}&limit=5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Found {len(data)} datasets for '{query}':")
        for ds in data:
            print(f"- {ds.get('id')} (Likes: {ds.get('likes')})")
    except Exception as e:
        print(f"Error searching HF: {e}")

if __name__ == "__main__":
    search_hf("musique")
