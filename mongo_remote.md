**Remote access to MongoDB**

- connect to control machine over ssh with the -X option
    - open firefox
    - go to 192.168.1.1 -> advanced -> advanced setup -> port forwarding
    - add rule for port 27017 to redirect traffic to analysis machine (currently 192.168.1.10)
- connect to analysis machine over ssh with the -X option
    - su -
    - firewall-config -> ports -> add 27017
    - check in /etc/mongod.cond: bindIP either commented or 0.0.0.0 to listen at all interfaces