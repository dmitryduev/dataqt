**The structure of the Robo-AO data archive**

```
/path/to/archive/
|--yyyymmdd/
   |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS/
   |  |--lucky/
   |  |  |--preview/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_full.png
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_cropped.png
   |  |  |--strehl/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_strehl.txt
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_box.fits
   |  |  |--pca/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.png
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.png
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.txt
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.fits
   |  |  |--_tentitavely_put_lucky_output_here_?
   |  |--faint/
   |  |  |--preview/
   |  |  |  |--...
   |  |  |--strehl/
   |  |  |  |--...
   |  |  |--pca/
   |  |  |  |--...
   |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_faint.fits
   |  |--planetary/
   |  |  |--preview/
   |  |  |  |--...
   |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_planetary.fits
   |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS.bz2
   |--.../
   |--seeing/
   |--calib/
|--.../
```