import pandas as pd
import configparser
import os

# read database credentials
group = snakemake.params["group"]
config = configparser.ConfigParser()
config.read(snakemake.input["db_credentials"])

# bash command to create table and restore tables from sql file
checkdb_cmd = "mysql -h " + config[group]["host"] + " -u " + config[group]["user"] + " -p" + config[group]["password"] + " -e \"use " + config[group]["database"] + "\""
create_cmd = "mysql -h " + config[group]["host"] + " -u " + config[group]["user"] + " -p" + config[group]["password"] + " -e \"CREATE DATABASE IF NOT EXISTS " + config[group]["database"] + ";\""
restore_cmd = "mysql -h " + config[group]["host"] + " -u " + config[group]["user"] + " -p" + config[group]["password"] + " " + config[group]["database"] + " < data/external/" + config[group]["database"] + ".sql"

try:
    os.system(checkdb_cmd)
except:
    print(config[group]["database"] + " DB already exists.")
else:
    os.system(create_cmd)
    os.system(restore_cmd)
