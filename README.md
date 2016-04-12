# dataqt
Robo-AO data monitoring service

- Clone the repository:

```
	git clone https://github.com/dmitryduev/dataqt.git
```

- Run Becky's PCA pipeline for a given date:

```
	python beckys.py path_to_pipelined_data path_to_psf_library path_to_pca_output_data --date YYYYMMDD --win W
```

- Run Becky's PCA pipeline for a single source:

```
	python pcapipe.py path_to_pipelined_source_data path_to_psf_library path_to_pca_output_data --win W
```

- Generate data to be shown on the website:

```
	python dataqt.py path_to_pipelined_data path_to_seeing_data path_to_pca_output_data --date YYYYMMDD
```

- Run the server using the pm2 process manager:

```
	pm2 start server_data_quality.py --interpreter=/path/to/python
```
