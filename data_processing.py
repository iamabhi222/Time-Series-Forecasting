import json
import re
from bs4 import BeautifulSoup
from collections import OrderedDict

def parse_debit_amounts(html_file):
    """Parses an HTML file to extract and sum debit amounts by date."""
    try:
        with open(html_file, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{html_file}' was not found.")
        return {}
    
    soup = BeautifulSoup(content, "html.parser")
    
    table = soup.find("table", class_="table table-bordered table-striped dateTable")
    if not table:
        # This error will show if the table itself isn't found
        print("Warning: Could not find the transaction table in the HTML file.")
        return {}
        
    debit_per_date = {}
    
    # --- THIS IS THE FIX ---
    # Instead of looking for a <tbody>, we find all <tr> tags
    # and use [1:] to skip the header row.
    rows = table.find_all("tr")
    if len(rows) < 2:
        print("Warning: No data rows found in the table.")
        return {}

    for row in rows[1:]: # Start from the second row to skip the header
        cells = row.find_all("td")
        if len(cells) < 6:
            continue
        
        date_text = cells[1].get_text(strip=True)
        debit_text = cells[5].get_text(strip=True)
        
        if debit_text == "-" or not debit_text:
            continue  # Skip credit rows or rows with no debit

        match = re.search(r"([\d,]+(?:\.\d+)?)", debit_text)
        if match:
            debit_value = float(match.group(1).replace(",", ""))
            debit_per_date[date_text] = debit_per_date.get(date_text, 0.0) + debit_value
            
    return debit_per_date

def update_json_file(json_path, new_data):
    """Loads, updates, and saves the JSON data, preventing duplicates."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    master_data = OrderedDict((item['date'], item) for item in existing_data)

    updated_count = 0
    new_count = 0
    for date, amount in new_data.items():
        if date in master_data:
            if master_data[date]['total_debit'] != amount:
                master_data[date]['total_debit'] = amount
                updated_count += 1
        else:
            master_data[date] = {"date": date, "total_debit": amount}
            new_count += 1

    final_data_list = list(master_data.values())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_data_list, f, indent=4)
        
    print(f"JSON update complete. Added: {new_count} new dates, Updated: {updated_count} existing dates.")


if __name__ == "__main__":
    HTML_FILE = "main.html"
    JSON_FILE = "data.json"
    
    daily_totals = parse_debit_amounts(HTML_FILE)
    
    if daily_totals:
        print(f"Successfully parsed {len(daily_totals)} unique dates from '{HTML_FILE}'.")
        update_json_file(JSON_FILE, daily_totals)
    else:
        print("No new data was parsed from the HTML file.")