import yfinance as yf
import os

# dapatkan data historis btc dari Yahoo Finance

# Ticker Bitcoin di Yahoo Finance
ticker = "BTC-USD"

# Rentang waktu data historis dari 17 Sep 2014 - hari ini
start_date = "2014-09-16"
end_date = "2024-11-01"

# Unduh data
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Membalik urutan data
data = data.iloc[::-1]

# Tambahkan kolom 'Date' sebagai kolom pertama
data.reset_index(inplace=True)  # Pindahkan index ke kolom agar 'Date' bisa jadi kolom pertama

# Urutkan kolom sesuai urutan yang diinginkan
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Tentukan folder penyimpanan
folder_path = "export"
file_path = os.path.join(folder_path, "BTC_USD_Historical_Data.csv")

# Buat folder jika belum ada
os.makedirs(folder_path, exist_ok=True)

# Simpan data ke CSV
data.to_csv(file_path, index=False)

# Cetak data untuk verifikasi
print(data)
