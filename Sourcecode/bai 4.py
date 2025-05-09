import pandas as pd
import numpy as np
import json
import os
import logging
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# File paths
INPUT_CSV = 'results.csv'
TRANSFER_CSV = 'transfer_values.csv'
PREDICT_CSV = 'transfer_predictions.csv'

# FootballTransfers URL
FOOTBALLTRANSFERS_URL = 'https://www.footballtransfers.com/en/search?q='

# Cache file for scraped values
CACHE_FILE = 'scrape_cache.json'

# Thiết lập logging
logging.basicConfig(filename='transfer_prediction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Đọc và tiền xử lý dữ liệu từ INPUT_CSV
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    # Chuẩn hóa tên cột
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_').str.replace('%', 'pct')
                  .str.replace('/', '_').str.replace('90', '_90'))
    # Chuyển đổi định dạng số
    for col in df.columns:
        if col not in ['player', 'team', 'position', 'nation']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    # Lọc cầu thủ phút > 900
    df = df[df['minutes'] > 900].copy()
    df['player'] = df['player'].str.strip()
    return df

# Chuyển text giá trị chuyển nhượng thành số (EUR)
def clean_value(val: str) -> float:
    if pd.isna(val) or not isinstance(val, str) or val.lower() in ['n/a', 'unknown', '']:
        return np.nan
    v = val.replace('€', '').replace('£', '').strip().lower()
    try:
        if 'm' in v:
            return float(v.replace('m', '')) * 1e6
        if 'k' in v:
            return float(v.replace('k', '')) * 1e3
        return float(v)
    except:
        return np.nan

# Khởi tạo WebDriver

def init_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument('--headless')
    ua = random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    ])
    options.add_argument(f'user-agent={ua}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    return driver

# Scrape chỉ từ FootballTransfers
def scrape_player_value(player: str, cache: dict, driver: webdriver.Chrome) -> str:
    if player in cache:
        return cache[player]
    try:
        ft_url = FOOTBALLTRANSFERS_URL + player.replace(' ', '+')
        driver.get(ft_url)
        result = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.market-value-amount'))
        )
        val = result.text
        logging.info(f"FT scraped {player}: {val}")
    except Exception as e:
        logging.warning(f"Failed to scrape {player}: {e}")
        val = 'N/a'
    cache[player] = val
    return val

# Thu thập giá trị chuyển nhượng cho toàn bộ DataFrame
def collect_transfer_values(df: pd.DataFrame) -> pd.DataFrame:
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE))
        except:
            cache = {}
    driver = init_driver()
    records = []
    for player in df['player']:
        raw = scrape_player_value(player, cache, driver)
        records.append({'player': player, 'transfer_value': clean_value(raw)})
        time.sleep(2)
    driver.quit()
    json.dump(cache, open(CACHE_FILE, 'w'))
    return pd.DataFrame(records)

# Chuẩn bị features và target cho mô hình
def prepare_features(df: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    data = df.merge(transfers, on='player', how='left')
    data['transfer_value'] = data['transfer_value'].fillna(data['transfer_value'].median())
    # Feature engineering
    data['goals_per_90'] = data['goals'] / (data['minutes'] / 90)
    data['assists_per_90'] = data['assists'] / (data['minutes'] / 90)
    # Lựa chọn features ban đầu
    feats = ['age', 'minutes', 'goals_per_90', 'assists_per_90', 'xg', 'xag',
             'prgc', 'prgp', 'prgr', 'sot_pct', 'sot_90', 'tkl', 'tklw', 'blocks',
             'touches', 'succ_pct_take_ons', 'fls', 'fld', 'aerial_won_pct']
    available = [f for f in feats if f in data.columns]
    # One-hot encode vị trí
    pos = pd.get_dummies(data['position'], prefix='pos')
    df_model = pd.concat([data[available], pos, data['transfer_value']], axis=1).dropna()
    return df_model

# Đào tạo và đánh giá mô hình

def train_and_evaluate(df_model: pd.DataFrame):
    X = df_model.drop(columns=['transfer_value'])
    y = df_model['transfer_value']
    # Impute và scale
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Hyperparameter tuning RandomForest
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, None]}
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                        scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    print(f"Best params: {grid.best_params_}")
    print(f"MSE: {mean_squared_error(y_test, preds):.2f}, R2: {r2_score(y_test, preds):.2f}")
    return best

if __name__ == '__main__':
    # Bước 1: Load dữ liệu
    df = load_data(INPUT_CSV)
    # Bước 2: Scrape giá trị chuyển nhượng
    transfers = collect_transfer_values(df)
    transfers.to_csv(TRANSFER_CSV, index=False, encoding='utf-8-sig')
    # Bước 3: Chuẩn bị dữ liệu cho mô hình
    df_model = prepare_features(df, transfers)
    # Bước 4: Đào tạo và đánh giá
    model = train_and_evaluate(df_model)
    # Bước 5 (tuỳ chọn): Dự đoán và lưu kết quả toàn bộ
    # preds_all = model.predict(...)
    # pd.DataFrame(...).to_csv(PREDICT_CSV, index=False, encoding='utf-8-sig')
