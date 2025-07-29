# 🇺🇳 UN INFO Joint Work Plans - Thematic Assistant

A secure, AI-driven assistant to explore programming insights from UN Cooperation Framework Joint Workplans (JWPs).

---

### 🌍 Our Mission
- This assistant supports evidence-based decision-making by making UN INFO 2024 programming data accessible and actionable.  
- It enables policy experts, UN staff, and researchers to interact with data-driven insights from Cooperation Framework Joint Workplans (JWPs), using an AI-powered chatbot and visualizations.

---

### 📚 Background
This tool was developed in preparation for the **DCO ASG Retreat** to:

- Analyze sub-output level programming data from **49+ UN Country Teams (UNCTs)**
- Cover six thematic areas:
  - **Youth**
  - **Education**
  - **Digital**
  - **Mining**
  - **Education in Crisis**
  - **IFF & Transnational Crime**
- Leverage UN INFO datasets uploaded and validated by UNCTs as of **April 2025**

The data comes from publicly available **Cooperation Framework (CF) JWPs** across Africa and includes both English and French content.

---

### 🛠 How It Works

The assistant runs on **Azure AI Foundry**, using enterprise-secure LLMs (OpenAI o1) within UN infrastructure. The process includes:

1. **Pre-processing:** Clean and structure open-ended programming data
2. **Prompt Engineering:** Apply thematic LLM prompts to extract insights
3. **Output Generation:** Summarize themes, challenges, and country examples
4. **Validation:** Reviewed with Microsoft Copilot and UN policy experts

All model interactions happen within a **private, role-secured environment**, ensuring compliance with **UN cybersecurity and data privacy protocols**.

---

### 💡 What You Can Do

- **Ask questions** via a chatbot on programming themes, funding gaps, or regional examples
- **Explore outputs** from real UNCT-reported data
- **View country-level visualizations** tied to funding or programming

---

### 🗂 Project Structure
```
uninfo_chatbot_z.zhao/
├── chatbot_app.py                 # Streamlit app with chat interface & theme selection
├── llm_analyzer.py                # Batch processing and generation using LLM for each theme
├── data_processor.py              # data processing steps
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and usage instructions
├── LICENSE                        # Licensing information
├── .gitignore                     # Git versioning exclusions
├── src/
│   ├── commonconst.py             # Central constants, theme mappings, paths
│   ├── data_processing.py         # Functions for loading & transforming thematic Excel data
│   ├── data/                      # Raw thematic Excel files organized by region
│   │   ├── Africa/
│   │   ├── Arab States/
│   │   ├── Asia Pacific/
│   │   ├── Europe and Central Asia/
│   │   └── Latin America and the Caribbean/
│   ├── outputs/
│   │   ├── 4o_outputs/            # LLM-generated insights (OpenAI o1 / Llama3)
│   │   └── data_outputs/          # Intermediate data for validation and review
```
---

📬 Contact Us

Developer: Zichen Zhao (Jackson)
Email: ziche.zhao@un.org
Organization: UN Development Coordination Office (UNDCO)