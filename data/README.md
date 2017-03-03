# Prepare Data
1. Download GIFs:<br>
```python main.py TAG FLAG```
* *TAG*: tag of GIFs you want to scrape, e.g. `cat`.<br>
* *FLAG*: -p: Match all given tags<br>
<tab><tab>-a: Match any given tags<br>
<tab><tab>-l LIMIT: Scraped GIFs limit<br>
2. Split GIFs into PNGs:<br>
<tab>Run 'python split.py *PATH TO GIFS*'