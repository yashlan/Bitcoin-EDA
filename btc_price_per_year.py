#membuat grafik harga bitcoin pertahun per tgl 31 desember
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Data harga Bitcoin pada 31 Desember dari 2009 hingga 2013. sumber: https://coincodex.com/article/31832/bitcoin-price-history/
data_manual = {
    "Year": [2009, 2010, 2011, 2012, 2013],
    "Price": [0.00099, 0.30, 4.72, 13.51, 749.24]
}

# Mengambil data harga Bitcoin menggunakan yfinance dengan periode max
def get_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    historical_data = btc.history(period="max")
    historical_data.reset_index(inplace=True)
    return historical_data

# Mengambil data
bitcoin_data = get_bitcoin_data()

# Filter data untuk tanggal 31 Desember dari setiap tahun
bitcoin_data["Date"] = pd.to_datetime(bitcoin_data["Date"])
bitcoin_data = bitcoin_data[bitcoin_data["Date"].dt.strftime("%m-%d") == "12-31"].copy()  # Menggunakan copy() untuk menghindari warning

# Ekstrak tahun dan harga
bitcoin_data["Year"] = bitcoin_data["Date"].dt.year
bitcoin_data = bitcoin_data[["Year", "Close"]]
bitcoin_data.rename(columns={"Close": "Price"}, inplace=True)

# Gabungkan data manual dan data dari Yahoo Finance
bitcoin_data_manual = pd.DataFrame(data_manual)
final_data = pd.concat([bitcoin_data_manual, bitcoin_data])

# Membagi data menjadi empat interval: 2009-2012, 2013-2016, 2017-2020, 2021-2024
intervals = [(2009, 2012), (2013, 2016), (2017, 2020), (2021, 2024)]

# Buat plot terpisah untuk setiap interval
for start_year, end_year in intervals:
    # Filter data untuk tahun dalam interval tersebut
    data_interval = final_data[(final_data["Year"] >= start_year) & (final_data["Year"] <= end_year)]
    
    # Plot harga Bitcoin pada 31 Desember setiap tahun untuk interval ini
    plt.figure(figsize=(5, 5))
    plt.bar(data_interval["Year"], data_interval["Price"], color="orange", label="Harga Bitcoin per 31 Desember")
    
    # Tambahkan label harga di setiap batang
    for year, price in zip(data_interval["Year"], data_interval["Price"]):
        price_label = f"${price}" if (start_year == 2009 and end_year == 2012) else f"${price:.2f}"
        plt.text(year, price + (max(data_interval["Price"]) * 0.01), price_label, fontsize=10, ha='center', va='bottom')

    plt.xticks(data_interval["Year"], rotation=0)
    plt.title(f"Harga Bitcoin ({start_year}-{end_year})", fontsize=13)
    plt.xlabel("Tahun", fontsize=14)
    plt.ylabel("Harga (USD)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print()

    # Menampilkan tabel data untuk interval ini
    # print(f"Data {start_year}-{end_year}:\n")
    # print(data_interval)
