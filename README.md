# Robo-AO data archive

This repository contains code that is used for the [Robo-AO](http://roboao.caltech.edu) automated data processing together with the (web-)tools to access the data.  This includes the pipeline to process faint target observations, estimate Strehl ratios, run PSF subtraction, generate contrast curves, and produce preview images for individual objects, and generate nightly summary plots with estimated seeing, 'joint' contrast curves, and Strehl ratios.  
>Robo-AO is the first automated laser guide star system that is currently installed on the Kitt Peak National Observatory's 2.1 meter telescope in Arizona. 

**archive.py** is the data processing engine.  
**server\_data\_archive.py** is the web-server for data access.

--- 

## How do I deploy the archiving system?

### Prerequisites
* pm2 process manager
* python libraries
  * flask
  * huey (Dima's forked version with a few tweaks)
  * mongoclient
  * image_registration (Dima's forked version with a few tweaks)
  ...

- Install fftw3
On mac:
```
brew install fftw
```
On Fedora:
```
yum install fftw3
```
- Install pyfftw (also see their github page for details) (use the right pip! (the one from anaconda)):
```
pip install pyfftw
```
- Clone image_registration repository from https://github.com/dmitryduev/image_registration.git
 I've made it use pyfftw by default, which is significantly faster than the numpy's fft,
 and quite faster (10-20%) than the fftw3 wrapper used in image_registration by default:
```
git clone https://github.com/dmitryduev/image_registration.git
```
- Install it:
```
python setup.py install --record files.txt
```
- To remove:
```
cat files.txt | xargs rm -rf
```


Clone the repository:
```bash
git clone https://github.com/dmitryduev/roboao-archive.git
```

---

### Configuration file (settings and paths)

* config.ini

---

### Set up and use MongoDB with authentication
Install MongoDB 3.2
(yum on Fedora; homebrew on MacOS)
On Mac OS use ```homebrew```. No need to use root privileges.
```
brew install mongodb
```
On Fedora, you would likely need to do these manipulation under root (```su -```)
 Create a file ```/etc/yum.repos.d/mongodb.repo```, add the following:  
```
[mongodb]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/7/mongodb-org/3.2/x86_64/ 
gpgcheck=0
enabled=1
```
 Install with yum:
```
yum install -y mongodb-org
```

Edit the config file. Config file location:  
```bash
/usr/local/etc/mongod.conf (Mac OS brewed)
/etc/mongod.conf (Linux)
```

Comment out:
```bash
#  bindIp: 127.0.0.1
```
Add: _(this is actually unnecessary)_
```bash
setParameter:
    enableLocalhostAuthBypass: true
```

Create (a new) folder to store the databases:
```bash
mkdir /Users/dmitryduev/web/mongodb/ 
```
In mongod.conf, replace the standard path with the custom one:
```bash
dbpath: /Users/dmitryduev/web/mongodb/
```

**On Mac (on Fedora, will start as a daemon on the next boot)**
Start mongod without authorization requirement:
```bash
mongod --dbpath /Users/dmitryduev/web/mongodb/ 
```
Connect to mongodb with mongo and create superuser (on Fedora, proceed as root):
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
Connect to mongodb (now not necessary as root)
```bash
mongo -u "admin" -p "roboaokicksass" --authenticationDatabase "admin" 
```
Add user to your database:
```bash
$ mongo
# This will create a databased called 'roboao' if it is not there yet
> use roboao
# Add user to your DB
> db.createUser(
    {
      user: "roboao",
      pwd: "roboaokicksass",
      roles: ["readWrite"]
    }
)
# Optionally create collections:
> db.createCollection("objects")
> db.createCollection("aux")
> db.createCollection("users")
# this will be later done from python anyways 
```
If you get locked out, start over (on Linux)
```bash
sudo service mongod stop
sudo service mongod start
```
To run the database manually (i.e. not as a service):
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
Check it out (optional):
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
# select database 'roboao'
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

**Use [Robomongo](https://robomongo.org) to display/edit DB data!! It's super handy!**  
Useful tip: check [this](https://docs.mongodb.com/manual/tutorial/enable-authentication/) out.

---

### Set up a Redis-based task queue to consume and process archiving jobs

Install the _huey_ task queue with 2 patches from DAD (see utils.py and consumer.py):
```bash
git clone https://github.com/dmitryduev/huey.git
cd huey
python setup.py install --record files.txt
```
Install redis-server if necessary.
Start Redis server on the standard port 6379 with pm2:
```bash
pm2 start redis-server -- --port 6379
```
In archive.py, make sure the Redis server is started with correct settings:
```python
from huey import RedisHuey
huey = RedisHuey(name='roboao.archive', host='127.0.0.1', port='6379', result_store=True)
```
(this should not raise any exceptions)

**With pm2, everything that's after '--' is passed to the script**

Start the task consumer with 4 parallel workers in the quiet mode polling stuff every 10 seconds without a crontab:
```bash
pm2 start huey_consumer.py --interpreter=/path/to/python -- /path/to/module.huey -k process -w 4 -d 10 -n -q
```
```
pm2 start huey_consumer.py --interpreter=/path/to/python -- /Users/dmitryduev/web/roboao-archive/archive.huey -k process -w 4 -d 10 -n -q
```

Check its status with ```pm2 status```. (saw errors a couple of times)

**It's a good idea to allocate ~half the number of the available cores**
_The Redis server and the task consumer are paused or stopped during daily nap time, which might be unnecessary_
```bash
pm2 stop redis-server
pm2 stop huey_consumer.py
```
start MongoDB (if not running already):
```bash
mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```

**Run the archiver!**
```python
python archive.py config.ini
```

### Data access via the web-server

Make sure to install python dependencies:
```
git clone https://github.com/pyvirtobs/pyvo.git
cd pyvo && /path/to/python setup.py install
pip install flask-login
```

Run the data access web-server using the pm2 process manager:
```bash
pm2 start server_data_archive.py --interpreter=/path/to/python
```

#### A short tutorial on how to use the web-site (once it's ready)

---

## Implementation details

* MongoDB noSQL database
* huey task queue + redis-server
* Flask back-end for the web tools

---

## How to work with the database

Mark all observations as not distributed (this will force):
```python
db.getCollection('objects').update({}, 
    { $set: 
        {'distributed.status': False,
         'distributed.last_modified': utc_now()}
    }, 
    {multi: true}
)
```

Force faint pipeline on a target:
```python
db.getCollection('objects').update_one({'_id': '4_351_Yrsa_VIC_lp600_o_20160925_110427.040912'}, 
    { $set: 
        {'pipelined.faint.status.force_redo': True,
         'pipelined.faint.last_modified': utc_now()}
    }
)
```

---

## Archive structure
The processed data are structured in the way described below. It should be straightforward to restore the database in case of a 'database disaster' keeping this structure in mind (in fact, **archive.py** will take care of that automatically once the database is up and running).

##### Science observations + daily summary plots (seeing, Strehl, contrast curves)
```
/path/to/archive/
├──yyyymmdd/
   ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS/
   │  ├──automated/
   │  │  ├──preview/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_full.png
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_cropped.png
   │  │  ├──strehl/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_strehl.txt
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_box.fits
   │  │  ├──pca/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.png
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.png
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.txt
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.fits
   │  │  └──_tentitavely_put_lucky_output_here_?
   │  ├──faint/
   │  │  ├──preview/
   │  │  │  └──...
   │  │  ├──strehl/
   │  │  │  └──...
   │  │  ├──pca/
   │  │  │  └──...
   │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_faint.fits
   │  ├──planetary/
   │  │  ├──preview/
   │  │  │  └──...
   │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_planetary.fits
   │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS.tar.bz2
   ├──.../
   ├──summary/
   │  ├──seeing/
   │  │  ├──yyyymmdd_hhmmss.png
   │  │  ├──...
   │  │  ├──seeing.yyyymmdd.txt
   │  │  └──seeing.yyyymmdd.png
   │  ├──contrast_curve.yyyymmdd.png
   │  └──strehl.yyyymmdd.png
   └──calib/?
└──.../
```
