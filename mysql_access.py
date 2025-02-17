import mysql.connector
from mysql.connector import Error, pooling

class Database:
    def __init__(self, host, database, user, password):
        """Initialize the database connection."""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(pool_name="mypool",
                                                               pool_size=5,
                                                               pool_reset_session=True,
                                                               host=host,
                                                               database=database,
                                                               user=user,
                                                               password=password)
            print("Database connection pool created successfully")
        except Error as e:
            print(f"Error while connecting to MySQL using connection pool: {e}")

    def get_connection(self):
        """Get a connection from the pool."""
        try:
            conn = self.connection_pool.get_connection()
            if conn.is_connected():
                print("Successfully retrieved connection from pool")
                return conn
        except Error as e:
            print(f"Error while retrieving connection from pool: {e}")

    def close_connection(self, conn):
        """Close the database connection."""
        if conn.is_connected():
            conn.close()
            print("MySQL connection is closed")

# Usage
if __name__ == '__main__':
    db_config = {
        'host': 'localhost',
        'database': 'wedding_planner',
        'user': 'root',
        'password': '1234'
    }
    db = Database(**db_config)

    # Example of getting a connection, using it and then closing it
    conn = db.get_connection()
    # Perform operations with 'conn' here, like executing SQL queries
    db.close_connection(conn)

