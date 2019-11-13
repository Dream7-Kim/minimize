import pymysql
import logging

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="test",
    charset="utf8"
)

print(conn)

cursor = conn.cursor()
sql = """SHOW DATABASES"""
cursor.execute(sql)
print(cursor.fetchall())

cursor.close()
