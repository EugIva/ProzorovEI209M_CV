import sys
import pandas as pd
import psycopg2 as db
import psycopg2.extras
import io

sys.path.append("/app/cv")

from utils import DBConnection


def get_connection(message):
    conn_opts = {
        'host': DBConnection.SQL_HOSTNAME,
        'port': DBConnection.SQL_PORT,
        'user': DBConnection.SQL_USERNAME,
        'password': DBConnection.SQL_PASSWORD,
        'dbname': DBConnection.SQL_MAIN_DATABASE
    }
    conn = db.connect(**conn_opts)
    if message is not None:
        print(message)

    return conn


def multy_insert(data, columns, schema, table, message='Connection completed'):
    conn = get_connection(message)
    df = pd.DataFrame(data, columns=columns)
    csv_io = io.StringIO()
    df.to_csv(csv_io, sep='\t', header=False, index=False)
    csv_io.seek(0)
    gp_cursor = conn.cursor()
    gp_cursor.execute(f"SET search_path = {schema}")
    gp_cursor.copy_from(csv_io, table)

    conn.commit()
    conn.close()


def execute_sql(sql, params, is_select=True, message=None):
    data = []
    conn = get_connection(message)
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            if is_select is True:
                data = cursor.fetchall()
            conn.commit()

    return data