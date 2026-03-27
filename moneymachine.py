import customtkinter as ctk
import yfinance as yf
import pandas as pd
import numpy as np
import threading

# --- The Logic Engine ---
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class MoneyMachinePro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Money Machine Pro v2.0")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Sidebar / Bench List
        self.bench = ["AMZN", "AAPL", "MSFT", "META", "GOOGL", "NVDA", "AMD", "PLTR", "TSLA", "NFLX"]
        
        # UI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar (Controls)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="MONEY\nMACHINE", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.pack(pady=20)

        self.search_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Manual Search...")
        self.search_entry.pack(pady=10, padx=20)
        
        self.search_btn = ctk.CTkButton(self.sidebar, text="Scan Ticker", command=self.manual_scan)
        self.search_btn.pack(pady=10, padx=20)

        self.refresh_btn = ctk.CTkButton(self.sidebar, text="Refresh Bench", fg_color="transparent", border_width=2, command=self.load_bench)
        self.refresh_btn.pack(pady=10, padx=20)

        # Main Dashboard
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#1a1a1a")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.dash_title = ctk.CTkLabel(self.main_frame, text="MARKET VELOCITY DASHBOARD", font=ctk.CTkFont(size=22, weight="bold"))
        self.dash_title.pack(pady=15)

        # Scrollable area for tickers
        self.scroll_frame = ctk.CTkScrollableFrame(self.main_frame, width=800, height=550, fg_color="transparent")
        self.scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Start loading bench immediately
        self.after(100, self.load_bench)

    def create_ticker_card(self, data):
        # Determine Color
        color = "#3fb950" if data['risk'] == "LOW" else "#d29922" if data['risk'] == "MED" else "#f85149"
        
        card = ctk.CTkFrame(self.scroll_frame, height=80, fg_color="#21262d", border_width=1, border_color="#30363d")
        card.pack(fill="x", pady=5, padx=5)

        # Ticker & Price
        ctk.CTkLabel(card, text=data['symbol'], font=ctk.CTkFont(size=18, weight="bold"), text_color="#58a6ff", width=80).pack(side="left", padx=20)
        ctk.CTkLabel(card, text=f"${data['price']:.2f}", font=ctk.CTkFont(size=16), width=100).pack(side="left", padx=10)
        
        # Risk Badge
        badge = ctk.CTkLabel(card, text=data['risk'], fg_color=color, text_color="white", corner_radius=6, width=70, font=ctk.CTkFont(size=12, weight="bold"))
        badge.pack(side="left", padx=20)

        # Strikes
        ctk.CTkLabel(card, text=f"90% Put: ${data['put']}  |  90% Call: ${data['call']}", font=("Segoe UI", 13), text_color="#8b949e").pack(side="left", padx=20)

        # Context
        ctk.CTkLabel(card, text=data['context'], font=("Segoe UI", 11, "italic"), text_color="#c9d1d9").pack(side="right", padx=20)

    def process_ticker(self, symbol):
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="3mo")
            if hist.empty: return None
            
            cp = hist['Close'].iloc[-1]
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            floor = hist['Close'].min()
            
            # 90% Math
            vol = np.std(hist['Close'].pct_change().dropna()) * np.sqrt(14)
            move = cp * (vol * 1.645)
            
            risk = "LOW" if cp > floor and cp > ma20 else "MED" if cp > floor else "HIGH"
            context = f"Above 20-MA" if cp > ma20 else f"Below 20-MA (${ma20:.0f})"

            return {
                "symbol": symbol, "price": cp, "risk": risk, 
                "put": round(cp - move), "call": round(cp + move), "context": context
            }
        except: return None

    def load_bench(self):
        # Clear existing
        for child in self.scroll_frame.winfo_children():
            child.destroy()
            
        def run():
            for symbol in self.bench:
                data = self.process_ticker(symbol)
                if data:
                    self.after(0, lambda d=data: self.create_ticker_card(d))
        
        threading.Thread(target=run).start()

    def manual_scan(self):
        symbol = self.search_entry.get().upper().strip()
        if symbol:
            data = self.process_ticker(symbol)
            if data:
                # Add manual result to top
                self.create_ticker_card(data)

if __name__ == "__main__":
    app = MoneyMachinePro()
    app.mainloop()