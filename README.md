# dataqt
Robo-AO data monitoring service

- Clone the repository:

```
	git clone https://github.com/dmitryduev/dataqt.git
```

- Generate data to be shown on the website:

```
	python dataqt.py path_to_pipelined_data path_to_seeing_data --date YYYYMMDD
```

- Run the server with the pm2 process manager:

```
	pm2 start server_data_quality.py --interpreter=/path/to/python
```
