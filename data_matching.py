from mysql_access import Database
from mysql.connector import Error

def fetch_category_data(db):
    """Fetch category id and name from the database."""
    try:
        # Get a connection from the pool
        conn = db.get_connection()
        cursor = conn.cursor()
        # SQL query to fetch category id and name
        cursor.execute("SELECT id, name FROM vendor_category")
        category_data = cursor.fetchall()
        return category_data
    except Error as e:
        print("Error while fetching category data:", e)
    finally:
        if conn.is_connected():
            db.close_connection(conn)

def create_category_mapping(category_data):
    """Create a dictionary mapping category IDs to names."""
    return {category_id: name for category_id, name in category_data}

# Main function to use the above functionalities
def main():
    # Database configuration details
    db_config = {
        'host': 'localhost',
        'database': 'wedding_planner',
        'user': 'root',
        'password': '1234'
    }

    # Create an instance of the Database class
    db = Database(**db_config)

    # Fetch category data
    category_data = fetch_category_data(db)

    # Create category mapping
    category_mapping = create_category_mapping(category_data)

    # Print or return the mapping
    print(category_mapping)

if __name__ == '__main__':
    main()
