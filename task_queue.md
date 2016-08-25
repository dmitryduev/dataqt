**Set up a Redis-based task queue to consume and process archiving jobs**

- Install the _huey_ task queue with 2 patches from DAD (see utils.py and consumer.py):
```
    git clone https://github.com/dmitryduev/huey.git
    python setup.py install --record files.txt
```

- Start Redis server on the standard port 6379 with pm2:
```
    pm2 start redis-server -- --port 6379
```

- In archive.py, make sure the Redis server is started with correct settings:
```
    huey = RedisHuey(name='roboao.archive', host='127.0.0.1', port='6379', result_store=True)
```

**With pm2, everything that's after '--' is passed to the script**

- Start the task consumer with 4 parallel workers in the quiet mode 
  polling stuff every 10 seconds without a crontab:
```
    pm2 start huey_consumer.py -- /path/to/module.huey -k process -w 4 -d 10 -n -q
    pm2 start huey_consumer.py -- /Users/dmitryduev/web/dataqt/archive.huey -k process -w 4 -d 10 -n -q
```

- The Redis server and the task consumer are paused or stopped during daily nap time
[This might be unnecessary]
```
    pm2 stop redis-server
    pm2 stop huey_consumer.py
```

- start mongo db
```
    mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```