# Prepare Data
1. Download GIFs:
run `python main.py *TAG FLAG*`. 
*TAG*: tag of GIFs you want to scrape, e.g. `cat`.
*FLAG*: -p: Match all given tags
		-a: Match any given tags
		-l LIMIT: Scraped GIFs limit
2. Split GIFs into PNGs:
run 'python split.py *PATH TO GIFS*'