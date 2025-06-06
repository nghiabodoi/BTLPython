import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math

# Đọc dữ liệu từ file results.csv
df = pd.read_csv('results.csv', encoding='utf-8-sig')

# Danh sách các cột thống kê từ code trước
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

# Lọc các cột số (numeric) để phân tích
numeric_columns = [col for col in STATS_COLUMNS if df[col].dtype in ['int64', 'float64'] or df[col].apply(lambda x: str(x).replace('N/a', 'nan').replace(',', '').replace('%', '').strip().replace('.', '', 1).isdigit() if str(x) != 'N/a' else False).all()]

# Chuyển đổi dữ liệu sang dạng số, thay 'N/a' bằng NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].replace('N/a', np.nan), errors='coerce')

# 1. Xác định top 3 cầu thủ có điểm cao nhất và thấp nhất cho mỗi thống kê
top_3_content = []
for col in numeric_columns:
    # Top 3 cao nhất
    top_high = df[['Player', col]].dropna().sort_values(by=col, ascending=False).head(3)
    top_3_content.append(f"Top 3 Highest {col}:")
    for idx, row in top_high.iterrows():
        top_3_content.append(f"  {row['Player']}: {row[col]}")
    
    # Top 3 thấp nhất
    top_low = df[['Player', col]].dropna().sort_values(by=col, ascending=True).head(3)
    top_3_content.append(f"Top 3 Lowest {col}:")
    for idx, row in top_low.iterrows():
        top_3_content.append(f"  {row['Player']}: {row[col]}")
    top_3_content.append("")  # Dòng trống để phân tách

# Lưu vào file top_3.txt
with open('top_3.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(top_3_content))
print("Saved top 3 players to top_3.txt")

# 2. Tính median, mean, và standard deviation cho mỗi thống kê
# Tạo DataFrame cho kết quả
teams = ['all'] + sorted(df['Team'].unique())
stats_summary = []

for team in teams:
    row_data = {'Team': team}
    if team == 'all':
        data = df
    else:
        data = df[df['Team'] == team]
    
    for col in numeric_columns:
        row_data[f'Median of {col}'] = data[col].median()
        row_data[f'Mean of {col}'] = data[col].mean()
        row_data[f'Std of {col}'] = data[col].std()
    
    stats_summary.append(row_data)

# Tạo DataFrame và thêm cột chỉ số
results2_df = pd.DataFrame(stats_summary)
results2_df.index = range(len(results2_df))
results2_df.index.name = None
results2_df.to_csv('results2.csv', index=True, index_label='', encoding='utf-8-sig')
print("Saved statistical summary to results2.csv")

# 3. Vẽ histogram cho phân phối của mỗi thống kê
output_dir = 'histograms'
os.makedirs(output_dir, exist_ok=True)

# Histogram riêng cho từng chỉ số
for col in numeric_columns:
    teams = sorted(df['Team'].unique())
    n_teams = len(teams)
    n_cols = 4  # 4 cột trong lưới subplot
    n_rows = math.ceil((n_teams + 1) / n_cols)  # +1 cho "All Players"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    # Histogram cho toàn bộ giải đấu
    axes[0].hist(df[col].dropna(), bins=20, edgecolor='black')
    axes[0].set_title(f'Distribution of {col} - All Players')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Histogram cho từng đội
    for i, team in enumerate(teams):
        team_data = df[df['Team'] == team][col].dropna()
        if not team_data.empty:
            axes[i + 1].hist(team_data, bins=20, edgecolor='black')
            axes[i + 1].set_title(f'Distribution of {col} - {team}')
            axes[i + 1].set_xlabel(col)
            axes[i + 1].set_ylabel('Frequency')
            axes[i + 1].grid(True, alpha=0.3)

    # Xóa các subplot không sử dụng
    for i in range(n_teams + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    safe_col = col.replace("/", "_").replace("%", "pct")
    plt.savefig(f'{output_dir}/{safe_col}_histogram.png')
    plt.close()
print(f"Saved individual statistic histograms to {output_dir}/")

# 4. Xác định đội có điểm cao nhất cho mỗi thống kê và phân tích đội tốt nhất
team_max_stats = {}
for col in numeric_columns:
    team_max = df.groupby('Team')[col].mean().idxmax()
    team_max_value = df.groupby('Team')[col].mean().max()
    team_max_stats[col] = (team_max, team_max_value)

# Đếm số lần mỗi đội đứng đầu
team_counts = pd.Series([team for stat, (team, value) in team_max_stats.items()]).value_counts()

# Phân tích đội tốt nhất
best_team = team_counts.idxmax()
best_team_count = team_counts.max()

# In kết quả ra console
print("Team with Highest Average Score for Each Statistic:")
for stat, (team, value) in team_max_stats.items():
    print(f"{stat}: {team} ({value:.2f})")
print("\nPerformance Analysis:")
print(f"The team {best_team} is likely performing the best in the 2024-2025 Premier League season, as it has the highest average score in {best_team_count} statistics, more than any other team.")