# Prepare Data
1. Download GIFs:<br>
run `python main.py *TAG FLAG*`. <br>
*TAG*: tag of GIFs you want to scrape, e.g. `cat`.<br>
*FLAG*: -p: Match all given tags<br>
		-a: Match any given tags<br>
		-l LIMIT: Scraped GIFs limit<br>
2. Split GIFs into PNGs:<br>
run 'python split.py *PATH TO GIFS*'