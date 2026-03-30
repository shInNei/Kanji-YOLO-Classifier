import sqlite3
import pandas as pd

class DataWarehouseManager:
    def __init__(self, db_path='data_warehouse.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def load_csv_to_warehouse(self, csv_path, table_name='Image_Metadata'):
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        print(f"Successfully loaded {len(df)} records into table '{table_name}'.")

    def execute_query(self, query):
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()

if __name__ == '__main__':
    dw = DataWarehouseManager()
    
    dw.load_csv_to_warehouse('Data_Lake/etl9g_metadata_master.csv')
    
    sql_query = """
        SELECT Image_ID, Source_File, Pixel_Mean 
        FROM Image_Metadata 
        WHERE Label_Unicode = '愛' 
        ORDER BY Pixel_Mean DESC 
        LIMIT 5;
    """
    
    result = dw.execute_query(sql_query)
    print("\nQuery Results:\n", result)
    
    dw.close()