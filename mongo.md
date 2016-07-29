# Set up mongodb with authentication

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
- Start mongod without authorization requirment:
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
            user :"someadmin",
            pwd: "secret", 
            roles: [{role: "userAdminAnyDatabase", db:"admin"}]})
    > exit 
```
- Connect to mongodb
```
   mongo -u "someadmin" -p "secret" --authenticationDatabase "admin" 
```
- Add user to your database:
```
    # Add user to your DB
    $ mongo
    > use some_db
    > db.createUser(
        {
          user: "mongouser",
          pwd: "someothersecret",
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
- Connect to database from pymongo:
```
    from pymongo import MongoClient
    client = MongoClient('ip_address_or_uri')
    db = client.some_db
    db.authenticate('mongouser', 'someothersecret')
```
Check it out:
```
    db['some_collection'].find_one()
```