from youtube_search import YoutubeSearch
import json

def get_youtube_results(query, limit=10):
    results = []
    try:
        search_results = YoutubeSearch(query, max_results=limit).to_dict()
        
        for video in search_results:
            results.append({
                'title': video['title'],
                'url': f"https://youtube.com{video['url_suffix']}",
                'duration': video['duration']
            })
        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
results = get_youtube_results("snapdragon elite new phones")
for video in results:
    print(f"Title: {video['title']}")
    print(f"URL: {video['url']}")
    print(f"Duration: {video['duration']}")
    print("---")