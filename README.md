# dataqt
Robo-AO data monitoring service

- Clone the repository:

```
	git clone https://github.com/dmitryduev/dataqt.git
```

- Settings and paths are stored in config.ini

- Run Becky's PCA pipeline for a given date:

```
	python beckys.py path_to_config_file [--date YYYYMMDD]
```

- Run Becky's PCA pipeline for a single source:

```
	python pcapipe.py path_to_pipelined_source_data path_to_psf_reference_library
                      path_to_psf_reference_library_short_names path_to_pca_output_data 
                      [--fwhm fwhm] [--win win] [--plsc plsc] [--sigma sigma]
                      [--nrefs nrefs] [--klip klip]
```

- Generate data to be shown on the website:

```
	python dataqt.py path_to_config_file [--date YYYYMMDD]
```

- Run the server using the pm2 process manager:

```
	pm2 start server_data_quality.py --interpreter=/path/to/python
```
