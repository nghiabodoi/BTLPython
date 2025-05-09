import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from io import StringIO
import os
import re
import uuid

# URLs and table IDs for different stat categories
table_links = {
    'Standard Stats': ('https://fbref.com/en/comps/9/stats/Premier-League-Stats', 'stats_standard'),
    'Shooting': ('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', 'stats_shooting'),
    'Passing': ('https://fbref.com/en/comps/9/passing/Premier-League-Stats', 'stats_passing'),
    'Goal and Shot Creation': ('https://fbref.com/en/comps/9/gca/Premier-League-Stats', 'stats_gca'),
    'Defense': ('https://fbref.com/en/comps/9/defense/Premier-League-Stats', 'stats_defense'),
    'Possession': ('https://fbref.com/en/comps/9/possession/Premier-League-Stats', 'stats_possession'),
    'Miscellaneous': ('https://fbref.com/en/comps/9/misc/Premier-League-Stats', 'stats_misc'),
    'Goalkeeping': ('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', 'stats_keeper')
}

# Required statistics as per assignment
STATS_COLUMNS = [
    "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Goals", "Assists", "Yellow Cards", "Red Cards",
    "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "GA90", "Save%", "CS%",
    "Penalty Kicks Save%",
    "SoT%", "SoT/90", "G/Sh", "Dist",
    "Passes Completed",
    "TkI", "TkIW",
    "Att Challenges", "Lost Challenges",
    "Blocks", "Sh Blocks", "Pass Blocks", "Int",
    "Touches", "Def Pen Touches", "Def 3rd Touches", "Mid 3rd Touches", "Att 3rd Touches", "Att Pen Touches",
    "Att Take-Ons", "Succ% Take-Ons", "Tkld% Take-Ons",
    "Carries", "ProDist", "ProgC Carries", "1/3 Carries", "CPA", "Mis", "Dis",
    "Rec", "PrgR Receiving",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerial Won", "Aerial Lost", "Aerial Won%"
]

# Column mappings from fbref table headers to required stats
COLUMN_MAPPINGS = {
    # Standard Stats
    'Nation': 'Nation', 'Squad': 'Team', 'Pos': 'Position', 'Age': 'Age',
    'Playing Time MP': 'Matches Played', 'Playing Time Starts': 'Starts', 'Playing Time Min': 'Minutes',
    'Performance Gls': 'Goals', 'Performance Ast': 'Assists', 'Performance CrdY': 'Yellow Cards', 'Performance CrdR': 'Red Cards',
    'Expected xG': 'xG', 'Expected xAG': 'xAG',
    'Progression PrgC': 'PrgC', 'Progression PrgP': 'PrgP', 'Progression PrgR': 'PrgR',
    'Per 90 Minutes Gls': 'Gls per 90', 'Per 90 Minutes Ast': 'Ast per 90', 
    'Per 90 Minutes xG': 'xG per 90', 'Per 90 Minutes xAG': 'xAG per 90',
    # Goalkeeping
    'Performance GA90': 'GA90', 'Performance Save%': 'Save%', 'Performance CS%': 'CS%',
    'Penalty Kicks Save%': 'Penalty Kicks Save%',
    # Shooting
    'Standard SoT%': 'SoT%', 'Standard SoT/90': 'SoT/90', 'Standard G/Sh': 'G/Sh', 'Standard Dist': 'Dist',
    # Passing
    'Total Cmp': 'Passes Completed',
    # Defensive Actions
    'Tackles Tkl': 'TkI', 'Tackles TklW': 'TkIW', 'Challenges Att': 'Att Challenges', 'Challenges Lost': 'Lost Challenges',
    'Blocks': 'Blocks', 'Blocks Sh': 'Sh Blocks', 'Blocks Pass': 'Pass Blocks', 'Int': 'Int',
    # Possession
    'Touches': 'Touches', 'Touches Def Pen': 'Def Pen Touches', 'Touches Def 3rd': 'Def 3rd Touches',
    'Touches Mid 3rd': 'Mid 3rd Touches', 'Touches Att 3rd': 'Att 3rd Touches', 'Touches Att Pen': 'Att Pen Touches',
    'Take-Ons Att': 'Att Take-Ons', 'Take-Ons Succ%': 'Succ% Take-Ons', 'Take-Ons Tkld%': 'Tkld% Take-Ons',
    'Carries': 'Carries', 'Carries PrgDist': 'ProDist', 'Carries PrgC': 'ProgC Carries',
    'Carries 1/3': '1/3 Carries', 'Carries CPA': 'CPA', 'Carries Mis': 'Mis', 'Carries Dis': 'Dis',
    'Receiving Rec': 'Rec', 'Receiving PrgR': 'PrgR Receiving',
    # Miscellaneous
    'Performance Fls': 'Fls', 'Performance Fld': 'Fld', 'Performance Off': 'Off', 'Performance Crs': 'Crs', 
    'Performance Recov': 'Recov',
    'Aerial Duels Won': 'Aerial Won', 'Aerial Duels Lost': 'Aerial Lost', 'Aerial Duels Won%': 'Aerial Won%'
}

def setup_driver():
    """Set up headless Chrome WebDriver."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def extract_player_data(name, url, table_id, driver):
    """Extract player data from a specific stat category page."""
    print(f'\nFetching data: {name}')
    driver.get(url)
    
    try:
        WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.ID, table_id)))
    except Exception as e:
        print(f'Loading table error {name}: {e}')
        return pd.DataFrame()
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', id=table_id)
    
    if not table:
        print(f'Unable to find table {name}')
        return pd.DataFrame()
    
    # Read table
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            group = col[0].strip() if col[0].strip() and 'Unnamed' not in col[0] else ''
            subgroup = col[1].strip() if col[1].strip() else col[0].strip()
            col_name = f"{group} {subgroup}" if group and group != subgroup else subgroup
            new_cols.append(col_name.strip())
        df.columns = new_cols
    else:
        df.columns = [col.strip() for col in df.columns]
    
    # Debug: Print columns
    print(f"Original columns {name}: {list(df.columns)}")
    
    # Find 'Player' column
    player_col = next((col for col in df.columns if 'player' in col.lower() and 'rk' not in col.lower()), None)
    if not player_col:
        print(f"ERROR: Can't find column 'Player' in {name}")
        return pd.DataFrame()
    
    # Clean data
    df = df.loc[df[player_col].notna() & (df[player_col] != player_col)].drop_duplicates(subset=player_col)
    df = df.rename(columns={player_col: 'Player'})
    
    # Select required columns for each table
    if name == 'Standard Stats':
        required_cols = ['Player', 'Nation', 'Squad', 'Pos', 'Age', 'Playing Time Min', 'Playing Time MP', 
                        'Playing Time Starts', 'Performance Gls', 'Performance Ast', 'Performance CrdY', 
                        'Performance CrdR', 'Expected xG', 'Expected xAG', 'Progression PrgC', 
                        'Progression PrgP', 'Progression PrgR', 'Per 90 Minutes Gls', 
                        'Per 90 Minutes Ast', 'Per 90 Minutes xG', 'Per 90 Minutes xAG']
    elif name == 'Shooting':
        required_cols = ['Player', 'Standard SoT%', 'Standard SoT/90', 'Standard G/Sh', 'Standard Dist']
    elif name == 'Passing':
        required_cols = ['Player', 'Total Cmp']
    elif name == 'Goal and Shot Creation':
        required_cols = ['Player']
    elif name == 'Defense':
        required_cols = ['Player', 'Tackles Tkl', 'Tackles TklW', 'Challenges Att', 'Challenges Lost',
                        'Blocks', 'Blocks Sh', 'Blocks Pass', 'Int']
    elif name == 'Possession':
        required_cols = ['Player', 'Touches', 'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd', 
                        'Touches Att 3rd', 'Touches Att Pen', 'Take-Ons Att', 'Take-Ons Succ%', 
                        'Take-Ons Tkld%', 'Carries', 'Carries PrgDist', 'Carries PrgC', 
                        'Carries 1/3', 'Carries CPA', 'Carries Mis', 'Carries Dis', 
                        'Receiving Rec', 'Receiving PrgR']
    elif name == 'Miscellaneous':
        required_cols = ['Player', 'Performance Fls', 'Performance Fld', 'Performance Off', 
                        'Performance Crs', 'Performance Recov', 'Aerial Duels Won', 
                        'Aerial Duels Lost', 'Aerial Duels Won%']
    elif name == 'Goalkeeping':
        required_cols = ['Player', 'Performance GA90', 'Performance Save%', 'Performance CS%', 
                        'Penalty Kicks Save%']
    else:
        required_cols = ['Player']
    
    # Filter available columns
    selected_cols = [col for col in required_cols if col in df.columns]
    if len(selected_cols) <= 1:  # Only 'Player'
        print(f"Can't find required columns for {name}, skipping")
        return pd.DataFrame()
    df = df[selected_cols]
    
    print(f"Selected columns {name}: {list(df.columns)}")
    return df

def main():
    # Set up Selenium driver
    driver = setup_driver()
    
    merged_df = pd.DataFrame()
    goalkeeping_df = pd.DataFrame()
    
    try:
        # Extract data from each table
        for name, (url, table_id) in table_links.items():
            df = extract_player_data(name, url, table_id, driver)
            if df.empty:
                continue
            if name == 'Goalkeeping':
                goalkeeping_df = df.copy()
            else:
                if merged_df.empty:
                    merged_df = df
                else:
                    df = df.drop(columns=[col for col in df.columns if col in merged_df.columns and col != 'Player'])
                    merged_df = pd.merge(merged_df, df, on='Player', how='outer')
    
        if merged_df.empty:
            print("No data collected.")
            return
        
        # Filter players with > 90 minutes
        min_col = 'Playing Time Min'
        if min_col in merged_df.columns:
            merged_df = merged_df[merged_df[min_col].notna()]
            try:
                merged_df[min_col] = merged_df[min_col].str.replace(',', '', regex=False).astype(float)
                merged_df = merged_df[merged_df[min_col] > 90]
                print(f"Found {len(merged_df)} players who played over 90 minutes")
            except Exception as e:
                print(f"Filtering Error 'Playing Time Min': {e}")
        else:
            print("ERROR: Can't find column 'Playing Time Min'")
        
        # Handle goalkeeping data
        if not goalkeeping_df.empty and not merged_df.empty:
            goalkeeping_df = goalkeeping_df[goalkeeping_df['Player'].isin(merged_df['Player'])]
            pos_col = 'Pos'
            if pos_col in merged_df.columns:
                goalkeeping_df['Pos'] = goalkeeping_df['Player'].map(merged_df.set_index('Player')[pos_col])
                for col in goalkeeping_df.columns:
                    if col not in ['Player', 'Pos']:
                        goalkeeping_df[col] = goalkeeping_df.apply(
                            lambda row: row[col] if pd.notna(row['Pos']) and 'GK' in row['Pos'] else 'N/a', 
                            axis=1
                        )
                goalkeeping_df.drop(columns='Pos', inplace=True)
                merged_df = pd.merge(merged_df, goalkeeping_df, on='Player', how='left')
            else:
                print("ERROR: Can't find column 'Pos'")
        
        # Extract first name for sorting
        merged_df['First Name'] = merged_df['Player'].apply(lambda x: x.split()[0] if x and isinstance(x, str) else 'Unknown')
        
        # Initialize output DataFrame with required columns including Player
        output_columns = ['Player'] + STATS_COLUMNS
        output_df = pd.DataFrame(index=merged_df.index, columns=output_columns)
        output_df.fillna('N/a', inplace=True)
        
        # Map data to required columns
        output_df['Player'] = merged_df['Player']  # Ensure Player column is filled
        for out_col in STATS_COLUMNS:
            if out_col in COLUMN_MAPPINGS.values():
                src_col = next((k for k, v in COLUMN_MAPPINGS.items() if v == out_col), None)
                if src_col and src_col in merged_df.columns:
                    output_df[out_col] = merged_df[src_col]
        
        # Clean up specific columns
        output_df['Nation'] = merged_df['Nation'].str.split().str[-1]  # Get country code
        output_df['Team'] = merged_df['Squad']
        output_df['Position'] = merged_df['Pos']
        output_df['Age'] = merged_df['Age'].str.split('-').str[0]  # Get age before hyphen
        
        # Verify no missing values remain
        missing_count = output_df.isna().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values found before saving. Filling with 'N/a'.")
            output_df.fillna('N/a', inplace=True)
        else:
            print("All cells filled, no missing values.")
        
        # Sort by first name
        output_df['First Name'] = merged_df['First Name']
        output_df = output_df.sort_values(by='First Name').drop(columns=['First Name'])
        
        # Save to CSV with UTF-8-SIG encoding to support special characters
        output_file = 'results.csv'
        try:
            output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f'Extracted file {output_file} successfully')
        except Exception as e:
            output_file = 'results_backup.csv'
            print(f'Error when saving to results.csv: {e}, try saving to {output_file}')
            output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f'Extracted file {output_file} successfully')
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()