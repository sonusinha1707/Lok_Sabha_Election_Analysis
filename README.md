# 🗳️ Lok Sabha Election Analysis 

A powerful Streamlit-based data analysis and visualization project that explores India's Lok Sabha election data. It allows users to interact with data on candidates, constituencies, parties, voter turnout, and gender-wise participation. The app also integrates Google's Gemini API to enable AI-powered data querying and insights.

![Banner](teacher2.png)

---

## 🚀 Features

- 📊 **Interactive Dashboard**: Visualize election data by year, state, constituency, party, and gender.
- 🧠 **AI-Powered Q&A**: Ask questions about election trends using Google Gemini (Gemini 1.5 Pro).
- 📍 **Geospatial Visualization**: Explore party-wise seat distribution with choropleth maps.
- 🧮 **Key Metrics**: Turnout percentage, total votes, constituencies, top parties, and more.
- 🔍 **Constituency & Candidate Drill-down**: Analyze vote distributions at granular levels.
- 💬 **Ask Gyan Bot**: Smart assistant that answers contextual questions using Gemini API.

---

## 📁 Project Structure

```bash
Lok_Sabha_Election_Analysis/
├── dashboard.py                # Main Streamlit app with visualizations + Gemini integration
├── Code_for_Data_Scrapping_.ipynb   # Jupyter notebook for scraping data
├── Data Cleaning All Tables.ipynb   # Notebook for cleaning datasets
├── Candidates.csv              # Candidate details by constituency
├── Election.csv                # Election-level results
├── constituency.csv            # Constituency-wise voter stats
├── gender_rat.csv              # Male vs female turnout by year
├── teacher2.png                # Custom mascot/image for UI
├── requirements.txt            # Python dependencies
└── .streamlit/, .devcontainer/ # Streamlit & dev configs
```

---

## 🧠 How It Works

### 🔎 Filters
- Year
- State
- Constituency
- Party

### 📌 Visuals
- Bar charts (votes, seats, category)
- Line graphs (voter trends)
- Pie charts (party-wise vote shares)
- Choropleth maps (state-level seats)
- Metric cards (votes, turnout, constituencies, etc.)

### 🤖 Gemini Integration
Ask data-specific questions or request detailed trend analysis with Gemini 1.5 Pro using custom prompts crafted for election data.

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/sonusinha1707/Lok_Sabha_Election_Analysis.git
cd Lok_Sabha_Election_Analysis
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Gemini API Key

Create a `.streamlit/secrets.toml` file with your key:

```toml
gemini_api = "your_gemini_api_key_here"
```

> 🔐 Don't expose this key in public repos.

---

## ▶️ Run the App

```bash
streamlit run dashboard.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push to your GitHub repo
2. Go to (https://loksabhaelectionanalysis.streamlit.app/)
3. Deploy your repo with `dashboard.py` as the entry point
4. Add your Gemini API key in the **Secrets Manager**

---

## 🙋‍♂️ Developed By

**Sonu Sinha**  
> Data Science Enthusiast | Streamlit Developer  
> 📧 sonusinha1707@gmail.com

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Show Your Support

If you liked this project:
- 🌟 Star this repo
- 🍴 Fork it
- 👨‍💻 Use it in your own election analysis
