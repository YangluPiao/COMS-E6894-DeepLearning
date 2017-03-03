# Prepare Data
1. Download GIFs:<br>
```python main.py TAG FLAG```<br>
* TAG: tag of GIFs you want to scrape, e.g. `cat`.<br>
* FLAG: -p: Match all given tags; -a: Match any given tags; -l LIMIT: Scraped GIFs limit<br>
2. Split GIFs into PNGs:<br>
```python split.py PATH_TO_GIFS```