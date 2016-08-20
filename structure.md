**The structure of the Robo-AO data archive**

```
/path/to/archive/
|__yyyymmdd/
   |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS/
      |__lucky/
      |   |__preview/
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_full.png
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_cropped.png
      |   |__strehl/
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_strehl.txt
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_box.fits
      |   |__pca/
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.png
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.png
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.txt
      |   |  |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.fits
      |   |___tentitavely_put_lucky_output_here_
      |__faint/
      |   |__preview/
      |   |  |__...
      |   |__strehl/
      |   |  |__...
      |   |__pca/
      |   |  |__...
      |   |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_faint.fits
      |__planetary/
      |   |__preview/
      |   |  |__...
      |   |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_planetary.fits
      |__programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS.bz2
   |__.../
   |__seeing/
|__.../
```