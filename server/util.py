import json

def get_countries():
    # Load the countries from the JSON file
    with open('artifacts/countries.json', 'r') as f:
        data = json.load(f)

    return data['countries']

if __name__ == "__main__":
    print(get_countries())