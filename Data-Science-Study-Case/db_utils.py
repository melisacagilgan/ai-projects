# Import necessary libraries
import sqlite3
import os
import pandas as pd


def create_db():
    """
    Creates a new SQLite database file named 'sales.db'. 

    If a file with the same name already exists, it will be deleted first 
    to ensure a fresh database file is created.

    Returns:
        None
    """

    # Define the database file name
    file_name = 'sales.db'

    # If the file already exists, remove it
    if os.path.exists(file_name):
        os.remove(file_name)

    # Create an empty database file
    open(file_name, 'w').close()

    # Print confirmation message
    print("Database named 'sales.db' created successfully.")


def connect_db():
    '''
    Establishes a connection to the SQLite database and returns the connection and cursor.
    Returns:
        conn: SQLite connection object
        cur: SQLite cursor object
    '''

    conn = sqlite3.connect('sales.db')
    cur = conn.cursor()

    return conn, cur


def commit_and_close(conn):
    '''
    Commits the current transaction and closes the database connection.
    Args:
        conn: SQLite connection object
    Returns: None
    '''

    conn.commit()
    conn.close()


def create_table():
    '''
    Creates the sales table in the database if it does not exist.
    Returns: None
    '''

    # Connect to database
    conn, cur = connect_db()

    # Create sales table if not exists
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            sales_date_id INTEGER NOT NULL
        )
        ''')

    # Commit the changes and close the connection
    commit_and_close(conn)

    print("Table created successfully.")


def insert_data(file_path):
    '''
    Inserts data from a Feather file into the sales table.

    Args:
        file_path (str): Path to the Feather file.
    Returns: None
    '''

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Connect to database
    conn, _ = connect_db()

    # Read the feather file and
    df = pd.read_feather(file_path)

    # Write the data from the DataFrame 'df' to the 'sales' table in the database
    df.to_sql('sales', conn, if_exists='append', index=False)

    # Commit the changes and close the connection
    commit_and_close(conn)

    print("Data inserted successfully from Feather file.")


def show_data():
    '''
    Returns all records from the sales table.

    Returns:
        list: List of tuples containing all records from the sales table.
    '''

    # Connect to database
    conn, cur = connect_db()

    # Retrieve all sales records, ordered by the most recent sales date first.
    cur.execute("SELECT * FROM sales ORDER BY sales_date_id DESC")
    data = cur.fetchall()

    # Determine column names
    columns = [description[0] for description in cur.description]

    # Create DataFrame for easier data manipulation
    df = pd.DataFrame(data, columns=columns)

    # Commit the changes and close the connection
    commit_and_close(conn)

    return df


def daily_sale_amount():
    '''
    Returns the total sales amount for each sales_date_id.

    Returns:
        list of tuples: Each tuple contains (sales_date_id, total_sale_amount)
    '''

    # Connect to database
    conn, cur = connect_db()

    # Execute the query to create a temporary view for daily sales summary
    cur.execute('''
        CREATE VIEW IF NOT EXISTS sale_amount AS
        SELECT
            sales_date_id,
            SUM(amount) AS total_sale_amount
        FROM sales
        GROUP BY sales_date_id
        ORDER BY sales_date_id DESC
    ''')

    # Retrieve all data from the newly created view, sorted by date (newest first)
    cur.execute('SELECT * FROM sale_amount ORDER BY sales_date_id DESC')
    results = cur.fetchall()

    # Commit the changes and close the connection
    commit_and_close(conn)

    return results


def daily_sale_count_by_sales_date():
    '''
    Returns the total number of sales for each sales_date_id.

    Returns:
        list of tuples: Each tuple contains (sales_date_id, sale_counter)
    '''

    # Connect to database
    conn, curr = connect_db()

    # Create a persistent database VIEW to aggregate daily sales counts
    curr.execute('''
        CREATE VIEW IF NOT EXISTS daily_sale_count AS
        SELECT
            sales_date_id,
            COUNT(amount) AS sale_counter
        FROM sales
        GROUP BY sales_date_id
        ORDER BY sales_date_id DESC
    ''')

    # Retrieve all aggregated daily sales count data from the newly created view
    curr.execute('SELECT * FROM daily_sale_count')
    results = curr.fetchall()

    # Commit the changes and close the connection
    commit_and_close(conn)

    return results


def daily_customer_sale_count_by_sales_date():
    '''
    Returns the total count of distinct customers for each sales_date_id.

    Returns:
        list of tuples: Each tuple contains (sales_date_id, customer_counter)
    '''

    # Connect to database
    conn, curr = connect_db()

    # Create a persistent database VIEW to aggregate the count of unique daily customers
    curr.execute('''
        CREATE VIEW IF NOT EXISTS customer_count AS
        SELECT
            sales_date_id,
            COUNT(DISTINCT customer_id) AS customer_counter
        FROM sales
        GROUP BY sales_date_id
        ORDER BY sales_date_id DESC
    ''')

    # Retrieve all aggregated unique customer count data from the newly created view
    curr.execute('SELECT * FROM customer_count')
    results = curr.fetchall()

    # Commit the changes and close the connection
    commit_and_close(conn)

    return results


def daily_sales_stats_by_date(specified_date):
    '''
    Calculates the overall Max, Average, and Median single transaction amount
    for the ENTIRE 20-day trailing window up to and including the specified date.

    Args:
        specified_date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with a single row containing the
                      sales_date, max, median, and average sales for the 20-day period.
    '''

    # Connect to the SQLite database
    conn, cur = connect_db()

    # Create a temporary table containing all transactions in the 20-day window
    cur.execute(f'''
        CREATE TEMPORARY TABLE amountView AS
        SELECT
            sales_date_id,
            amount
        FROM sales
        WHERE DATE(
                substr(CAST(sales_date_id AS TEXT),1,4) || '-' ||
                substr(CAST(sales_date_id AS TEXT),5,2) || '-' ||
                substr(CAST(sales_date_id AS TEXT),7,2))
            BETWEEN DATE(?, '-20 days') AND DATE(?)
    ''', (specified_date, specified_date))

    # Fetch all rows from the temporary table to check the data
    cur.execute('SELECT * FROM amountView')

    # Calculate the running maximum amount for each row ordered by sales_date_id
    cur.execute(
        ''' SELECT MAX(amount) OVER ()
            FROM amountView
        '''
    )
    max_val = cur.fetchone()[0]

    # Calculate the average amount across the 20-day window
    cur.execute(
        ''' SELECT AVG(amount)
            FROM amountView
        '''
    )
    avg_val = cur.fetchone()[0]

    # Attempt to calculate the median (currently this is a placeholder)
    cur.execute(
        ''' SELECT amount
            FROM amountView
            ORDER BY amount DESC
        '''
    )
    results = cur.fetchall()

    # The dataset size (n=20) is even, so the median is the average of the two middle values.
    # Calculate the position of the first middle element which is the 10th element
    f_index = int(20/2)-1

    # Calculate the position (1-based index) of the second middle element which is the 11th element
    s_index = int(20/2)

    f_middle = results[f_index][0]
    s_middle = results[s_index][0]

    # Calculate the Median: the arithmetic mean (average) of the two middle values.
    median_val = (f_middle+s_middle)/2

    # Commit any changes and close the connection
    commit_and_close(conn)

    # Convert the fetched results into a single-row DataFrame
    df = pd.DataFrame({
        'sales_date': [specified_date],
        'max': [max_val],
        'median': [median_val],
        'average': [avg_val]
    })

    return df


def monthly_sales_amount_per_customer():
    '''
    Returns the total sales amount per customer for each month.

    Returns:
        list of tuples: Each tuple contains(sales_month, customer_id, monthly_total_amount)
    '''

    # Connect database
    conn, cur = connect_db()

    # Execute the query to calculate the total sales amount for each customer, grouped by month
    cur.execute('''
        SELECT
            substr(sales_date_id, 1, 6) AS sales_month,
            customer_id,
            SUM(amount) AS monthly_total_amount
        FROM sales
        GROUP BY sales_month, customer_id
        ORDER BY sales_month DESC
    ''')
    results = cur.fetchall()

    # Commit the changes and close the connection
    commit_and_close(conn)

    return results


def monthly_sales_counter_per_customer():
    '''
    Returns the total number of sales per customer for each month.

    Returns:
        list of tuples: Each tuple contains(sales_month, customer_id, monthly_sales_count)
    '''

    # Connect to database
    conn, cur = connect_db()

    # Execute the query to count the total number of transactions (sales) for each customer, grouped by month
    cur.execute('''
        SELECT
            substr(sales_date_id, 1, 6) AS sales_month,
            customer_id,
            COUNT(DISTINCT id) AS monthly_sales_count
        FROM sales
        GROUP BY sales_month, customer_id
        ORDER BY sales_month DESC
    ''')
    results = cur.fetchall()

    # Commit the changes and close the connection
    commit_and_close(conn)

    return results
