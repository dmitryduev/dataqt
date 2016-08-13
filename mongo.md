# Set up and use mongodb with authentication

Also check https://docs.mongodb.com/manual/tutorial/enable-authentication/

- Install (yum on Fedora; homebrew on MacOS)
- Edit the config file. Config file location:
```
    /usr/local/etc/mongod.conf (Mac OS brewed)
    /etc/mongodb.conf (Linux)
```
Comment out:
```
    #  bindIp: 127.0.0.1
```
Add:
```
    setParameter:
      enableLocalhostAuthBypass: true
```
- Start mongod without authorization requirement:
```
   mongod --dbpath /Users/dmitryduev/web/mongodb/ 
```
- Connect to mongodb with mongo and create superuser:
```
    # Create your superuser
    $ mongo
    > use admin
    > db.createUser(
        {
            user :"admin",
            pwd: "roboaokicksass", 
            roles: [{role: "userAdminAnyDatabase", db:"admin"}]})
    > exit 
```
- Connect to mongodb
```
   mongo -u "admin" -p "roboaokicksass" --authenticationDatabase "admin" 
```
- Add user to your database:
```
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
- If you get locked out, start over (on Linux)
```
    sudo service mongod stop
    sudo mv /data/admin.* .  # for backup
    sudo service mongod start
```
- To run manually (i.e. not as a service):
```
    mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```
- Connect to database from pymongo:
```
    from pymongo import MongoClient
    client = MongoClient('ip_address_or_uri')
    db = client.roboao
    db.authenticate('roboao', 'roboaokicksass')
```
Check it out:
```
    db['some_collection'].find_one()
```

**Add admin user for data access on the website**

- Connect to database from pymongo and do an insertion:
```
    from pymongo import MongoClient
    from werkzeug.security import generate_password_hash
    import datetime
    client = MongoClient('ip_address_or_uri')
    db = client.roboao
    db.authenticate('roboao', 'roboaokicksass')
    coll = db['users']
    result = coll.insert_one(
            {'_id': 'adminus',
             'password': generate_password_hash('robo@0'),
             'programs': 'all',
             'last_modified': datetime.datetime.now()}
    )
```

** Use Robomongo to display DB data!! It's super handy! **