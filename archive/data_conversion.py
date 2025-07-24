import json
import csv
import os
from pathlib import Path

def convert_json_to_csv(input_dir="data/initial_questions", output_dir=None):
    """
    Convert all JSON files in input_dir to CSV format with columns: type, question, answer
    
    Args:
        input_dir (str): Directory containing JSON files to process
        output_dir (str): Directory to save CSV files (if None, saves alongside JSON files)
    """
    
    # Create Path objects for easier file handling
    input_path = Path(input_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return
    
    # Set output directory (default to same as input)
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the directory
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            # Read and parse JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract questions and answers
            questions = data.get('question', {})
            answers = data.get('answer', {})
            
            # Prepare CSV data
            csv_data = []
            
            # Process each question-answer pair
            for key in questions.keys():
                question_text = questions[key]
                answer_text = answers.get(key, "None")  # Default to "None" if answer missing
                
                csv_data.append({
                    'type': 'UNTYPED',
                    'question': question_text,
                    'answer': answer_text
                })
            
            # Create output CSV filename
            csv_filename = json_file.stem + '.csv'
            csv_filepath = output_path / csv_filename
            
            # Write to CSV
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['type', 'question', 'answer']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                writer.writerows(csv_data)
            
            print(f"✓ Converted {json_file.name} -> {csv_filename} ({len(csv_data)} rows)")
            
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing {json_file.name}: Invalid JSON - {e}")
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
    
    print("Conversion complete!")

def main():
    """
    Main function to run the conversion
    """
    # You can customize these paths as needed
    input_directory = "data/initial_questions"
    
    # Optional: specify a different output directory
    # output_directory = "data/csv_output"
    output_directory = None  # This will save CSVs in the same directory as JSON files
    
    convert_json_to_csv(input_directory, output_directory)

if __name__ == "__main__":
    main()