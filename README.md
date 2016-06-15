# dataqt
Robo-AO data monitoring service

- Clone the repository:

```
	git clone https://github.com/dmitryduev/dataqt.git
```

- Settings and paths are stored in config.ini

- Run the PCA pipeline for a given date (default - today):

```
	python beckys.py path_to_config_file [--date YYYYMMDD]
```

- Run the PCA pipeline for a single source:

```
	python pcapipe.py path_to_pipelined_source_data path_to_psf_reference_library
                      path_to_psf_reference_library_short_names path_to_pca_output_data 
                      [--fwhm fwhm] [--win win] [--plsc plsc] [--sigma sigma]
                      [--nrefs nrefs] [--klip klip]
```

- Run the Strehl ratio calculator for a given date (default - today):

```
	python maissas.py path_to_config_file [--date YYYYMMDD]
```

- Run the seeing estimation code for a given date (default - today):

```
	python seeing.py path_to_config_file [--date YYYYMMDD]
```

- Generate data to be shown on the website for a given date (default - today):

```
	python dataqt.py path_to_config_file [--date YYYYMMDD]
```

- Run the server using the pm2 process manager:

```
	pm2 start server_data_quality.py --interpreter=/path/to/python
```
