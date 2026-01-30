import argparse
import fastavro
import csv
import os
import sys

def convert_avro_to_csv(avro_path, csv_path):
    print(f"Reading from: {avro_path}")
    try:
        with open(avro_path, 'rb') as f_in:
            reader = fastavro.reader(f_in)
            
            # Try to get fields from schema if available
            fields = []
            if reader.writer_schema and 'fields' in reader.writer_schema:
                fields = [f['name'] for f in reader.writer_schema['fields']]
            
            record_count = 0
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = None
                
                for i, record in enumerate(reader):
                    if i == 0:
                        if not fields:
                            fields = list(record.keys())
                        
                        writer = csv.DictWriter(f_out, fieldnames=fields)
                        writer.writeheader()
                    
                    if writer:
                        writer.writerow(record)
                    record_count += 1
            
            print(f"Success! Converted {record_count} records.")
            print(f"Output saved to: {csv_path}")
            
    except FileNotFoundError:
        print(f"Error: The file '{avro_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Avro files to CSV.")
    parser.add_argument("input_file", help="Path to the input .avro file")
    parser.add_argument("-o", "--output", help="Path to the output .csv file (optional)")

    args = parser.parse_args()
    
    input_path = args.input_file
    output_path = args.output
    
    if not output_path:
        # Default to same name with .csv extension
        base, _ = os.path.splitext(input_path)
        output_path = base + ".csv"
    
    convert_avro_to_csv(input_path, output_path)
