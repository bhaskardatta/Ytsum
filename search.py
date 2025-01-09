from youtubesearchpython import VideosSearch

search = VideosSearch('snapdragon elite new phones', limit=10)
results = search.result()['result']
for video in results:
    print(video['title'], video['link'])