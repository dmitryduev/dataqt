# Robo-AO data archive

This repository contains code that is used for the [Robo-AO](roboao.caltech.edu) automated data processing together with the (web-)tools to access the data.  This includes the pipeline to process faint target observations, estimate Strehl ratios, run PSF subtraction, generate contrast curves, and produce preview images for individual objects, and generate nightly summary plots with estimated seeing, 'joint' contrast curves, and Strehl ratios.  
>Robo-AO is the first automated laser guide star system that is currently installed on the Kitt Peak National Observatory's 2.1 meter telescope in Arizona. 

**archive.py** is the data processing engine.  
**server_data_archive.py** is the web-server for data access.

--- 

## How do I deploy the archiving system?

### Prerequisites
* pm2
* python libraries
  * flask
  * huey
  * mongoclient
  ...

---

### Configuration file

* config.ini

---

### Set up and use MongoDB with authentication
Install (yum on Fedora; homebrew on MacOS)
Edit the config file. Config file location:  
```bash
/usr/local/etc/mongod.conf (Mac OS brewed)
/etc/mongodb.conf (Linux)
```
Comment out:
```bash
#  bindIp: 127.0.0.1
```
Add:
```bash
setParameter:
    enableLocalhostAuthBypass: true
```
Start mongod without authorization requirement:
```bash
mongod --dbpath /Users/dmitryduev/web/mongodb/ 
```
Connect to mongodb with mongo and create superuser:
```bash
# Create your superuser
$ mongo
> use admin
> db.createUser(
    {
        user: "admin",
        pwd: "roboaokicksass", 
        roles: [{role: "userAdminAnyDatabase", db: "admin"}]})
> exit 
```
Connect to mongodb
```bash
mongo -u "admin" -p "roboaokicksass" --authenticationDatabase "admin" 
```
Add user to your database:
```bash
# Add user to your DB
$ mongo
> use some_db
> db.createUser(
    {
      user: "roboao",
      pwd: "roboaokicksass",
      roles: ["readWrite"]
    }
)
```
If you get locked out, start over (on Linux)
```bash
sudo service mongod stop
sudo mv /data/admin.* .  # for backup
sudo service mongod start
```
To run manually (i.e. not as a service):
```bash
mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```
Connect to database from pymongo:
```python
from pymongo import MongoClient
client = MongoClient('ip_address_or_uri')
db = client.roboao
db.authenticate('roboao', 'roboaokicksass')
```
Check it out:
```python
db['some_collection'].find_one()
```
#### Add admin user for data access on the website
Connect to database from pymongo and do an insertion:
```python
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import datetime
client = MongoClient('ip_address_or_uri')
db = client.roboao
db.authenticate('roboao', 'roboaokicksass')
coll = db['users']
result = coll.insert_one(
        {'_id': 'admin',
         'password': generate_password_hash('robo@0'),
         'programs': 'all',
         'last_modified': datetime.datetime.now()}
)
```

**Use [Robomongo](robomongo.org) to display DB data!! It's super handy!**
Useful tip: check [this](https://docs.mongodb.com/manual/tutorial/enable-authentication/) out.
---

### Set up a Redis-based task queue to consume and process archiving jobs

Install the _huey_ task queue with 2 patches from DAD (see utils.py and consumer.py):
```bash
git clone https://github.com/dmitryduev/huey.git
python setup.py install --record files.txt
```
Start Redis server on the standard port 6379 with pm2:
```bash
pm2 start redis-server -- --port 6379
```
In archive.py, make sure the Redis server is started with correct settings:
```python
    huey = RedisHuey(name='roboao.archive', host='127.0.0.1', port='6379', result_store=True)
```

**With pm2, everything that's after '--' is passed to the script**

Start the task consumer with 4 parallel workers in the quiet mode polling stuff every 10 seconds without a crontab:
```bash
pm2 start huey_consumer.py -- /path/to/module.huey -k process -w 4 -d 10 -n -q
pm2 start huey_consumer.py -- /Users/dmitryduev/web/dataqt/archive.huey -k process -w 4 -d 10 -n -q
```

**It's a good idea to allocate ~half the number of the available cores**
_The Redis server and the task consumer are paused or stopped during daily nap time, which might be unnecessary_
```bash
pm2 stop redis-server
pm2 stop huey_consumer.py
```
start MongoDB:
```bash
mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```
---

## Implementation details
* MongoDB noSQL database
* huey task queue + redis-server
* Flask back-end for the web tools
---

## Archive structure
The processed data are structured in the way described below. It should be straightforward to restore the database in case of a 'database disaster' keeping this structure in mind (in fact, **archive.py** will take care of that automatically once the database is up and running).

##### Science observations + daily summary plots (seeing, Strehl, contrast curves)
```
/path/to/archive/
|--yyyymmdd/
   |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS/
   |  |--automated/
   |  |  |--preview/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_full.png
   |  |  |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_cropped.png
   |  |  |--strehl/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_strehl.txt
   |  |  |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_box.fits
   |  |  |--pca/
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.png
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.png
   |  |  |  |--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.txt
   |  |  |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.fits
   |  |  `--_tentitavely_put_lucky_output_here_?
   |  |--faint/
   |  |  |--preview/
   |  |  |  `--...
   |  |  |--strehl/
   |  |  |  `--...
   |  |  |--pca/
   |  |  |  `--...
   |  |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_faint.fits
   |  |--planetary/
   |  |  |--preview/
   |  |  |  `--...
   |  |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_planetary.fits
   |  `--programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS.tar.bz2
   |--.../
   |--summary/
   |  |--seeing/
   |  |  |--yyyymmdd_hhmmss.png
   |  |  |--...
   |  |  |--seeing.yyyymmdd.txt
   |  |  `--seeing.yyyymmdd.png
   |  |--contrast_curve.yyyymmdd.png
   |  `--strehl.yyyymmdd.png
   `--calib/?
|--.../
```