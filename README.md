# ระบบวิเคราะห์และคาดการณ์ความเสี่ยงการผิดนัดชำระหนี้ด้วย Machine Learning & Explainable AI
### AI-Driven Credit Risk Prediction & Transparent Underwriting Intelligence System

> 🎓 โครงงานรายวิชา **240-403 เตรียมสหกิจศึกษา** ภาคการศึกษา 1/2569  
> 🏛️ **หลักสูตรวิศวกรรมปัญญาประดิษฐ์และวิทยาการข้อมูล สาขาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยสงขลานครินทร์ (PSU)**  
> 👩‍💻 **ผู้เสนอโครงงาน:** นางสาวยลรดี ภักดีจิตร์ (รหัสนักศึกษา 6610110734)  
> 👨‍🏫 **อาจารย์ที่ปรึกษาสหกิจ:** ผศ.ดร. สหพงศ์ สมวงค์  

---

## ⚡ โหลดลงมาแล้วใช้งานได้ทันที (1-Click Quickstart)

เมื่อทำการ Clone หรือ Download โครงการนี้มาแล้ว สามารถเริ่มใช้งานได้ทันที 2 วิธี:

### วิธีที่ 1: ดับเบิลคลิกไฟล์สคริปต์ (สำหรับ Windows)
1. ดับเบิลคลิก **`setup.bat`** (ระบบจะสร้าง Virtual Environment และติดตั้งไลบรารีทั้งหมดให้โดยอัตโนมัติ)
2. ดับเบิลคลิก **`run_app.bat`** เพื่อเปิดหน้าเว็บ **Credit Risk Executive Dashboard** บนเบราว์เซอร์ (`http://localhost:8501`)
3. (ทางเลือก) ดับเบิลคลิก **`run_api.bat`** เพื่อเปิดบริการ **FastAPI REST Service** (`http://localhost:8000/docs`)

---

### วิธีที่ 2: รันผ่าน Command Line (Windows / macOS / Linux)

```bash
# 1. โคลนคลังโค้ด
git clone <YOUR_REPOSITORY_URL>
cd credit_risk

# 2. สร้างและเปิดใช้งาน Virtual Environment
# บน Windows:
python -m venv .venv
.\.venv\Scripts\activate
# บน macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# 3. ติดตั้ง Dependencies ทั้งหมด
pip install -r requirements.txt

# 4. รันระบบ Dashboard หลัก
streamlit run app.py
```
เปิดเบราว์เซอร์ไปที่ **`http://localhost:8501`** เพื่อเข้าใช้งานระบบได้ทันที (โมเดลทั้งหมดถูกเทรนและบันทึกไว้ใน `models/` พร้อมใช้งาน ไม่ต้องเสียเวลารอเทรนใหม่)

---

## 🏛️ สถาปัตยกรรมระบบขั้นสูงและฟีเจอร์ระดับสถาบันการเงิน (Institutional FinTech Architecture)

ระบบนี้ได้รับการออกแบบตามมาตรฐานสากลด้านการกำกับดูแลความเสี่ยงสถาบันการเงิน (Basel Risk Governance & Auditable AI) ประกอบด้วยความสามารถหลัก:

1. **ระบบประเมินความเสี่ยงและมาตรวัด Real-Time (Credit Underwriting & Scoring):**
   - คำนวณคะแนนความเสี่ยง **Credit Risk Score (0 - 100 pt)**
   - จัดกลุ่มระดับความเสี่ยง 3 ระดับตาม Section 7.3 ในข้อเสนอโครงงาน:
     - 🟢 **Low Risk (0 - 30 pt):** ควรอนุมัติสินเชื่อ (Approve)
     - 🟡 **Medium Risk (31 - 70 pt):** ควรพิจารณาเพิ่มเติมหรือขอเอกสารเสริม (Under Review)
     - 🔴 **High Risk (71 - 100 pt):** ปฏิเสธสินเชื่อ (Reject)
   - สร้าง **Official Credit Underwriting Memorandum** แบบฟอร์มบันทึกการพิจารณาที่มีตราประทับพร้อมบล็อกลายเซ็นอนุมัติ
   - **ดาวน์โหลดเอกสาร PDF (1-Click Official Dossier):** ส่งออกรายงานการประเมินสินเชื่อฉบับทางการด้วย ReportLab

2. **ระบบจำลองการปรับโครงสร้างสินเชื่อ (What-If / Counterfactual Actionable Recourse):**
   - สำหรับผู้สมัครที่อยู่ในกลุ่มความเสี่ยงปานกลางหรือสูง ระบบจะคำนวณวงเงินกู้สูงสุดที่แนะนำให้อัตโนมัติ เพื่อให้ผู้สมัครลดความเสี่ยงลงมาอยู่ในเกณฑ์ **Low Risk (≤ 30 pt)**
   - แถบปรับค่าตัวแปรแบบ Interactive (วงเงินกู้, อัตราดอกเบี้ย, รายได้เสริม) ที่คำนวณผลกระทบต่อคะแนนความเสี่ยงทันที

3. **การปรับจุดตัดการตัดสินใจตามต้นทุนทางธุรกิจ (Cost-Sensitive Threshold Optimization):**
   - ในธุรกิจสินเชื่อ หนี้เสีย (**False Negative**) สร้างความเสียหายสูงกว่าการปฏิเสธลูกค้าดี (**False Positive**) ประมาณ 5-20 เท่า
   - โมเดลได้วิเคราะห์จุดตัดที่เหมาะสมที่สุด (**Optimal Threshold = 0.26**) แทนจุดตัดทั่วไป (0.50) ช่วยลดหนี้เสียที่หลุดรอดลงจาก 408 ราย เหลือ **292 ราย** และลดความเสียหายคาดหวังลงได้ถึง **8.85%**

4. **ระบบจัดเก็บประวัติและการตรวจสอบย้อนหลัง (Database Audit Logging & Portfolio KPIs):**
   - บันทึกผลการพิจารณาและข้อมูลผู้สมัครทุกลงฐานข้อมูล SQLite (`credit_risk.db`) ตาม Section 9.2.3 ใน Proposal
   - แสดงสถิติภาพรวม Portfolio: จำนวนเคสที่ประเมิน, คะแนนเฉลี่ย, สัดส่วนการอนุมัติ, และปุ่มส่งออก Audit Trail เป็น CSV

5. **บริการ REST API Backend (FastAPI Service):**
   - มีไฟล์ `api.py` ให้บริการ Endpoint มาตรฐานตามภาพที่ 9.2 ใน Proposal (`POST /predict`, `POST /what-if`, `GET /metrics`, `GET /assessments`)

6. **ระบบหน้าต่างวิเคราะห์ขั้นสูง (Multi-Page Sidebar):**
   - 📉 **Basel III Macroeconomic Stress Testing:** จำลองวิกฤตเศรษฐกิจ 3 สถานการณ์ (Mild Downturn, Severe Recession, Stagflation Shock)
   - ⚖️ **AI Fairness & Non-Discrimination Audit:** ตรวจสอบความยุติธรรมตามเกณฑ์ Four-Fifths Rule (Disparate Impact Ratio)
   - 📡 **API Playground:** หน้าต่างทดสอบการยิง REST API แบบ Interactive

---

## 📊 ผลการทดสอบเปรียบเทียบโมเดล (Benchmark Results)

| โมเดล (Algorithm) | ROC-AUC | F1-Score | Accuracy | Recall (ตรวจจับหนี้เสีย) | Brier Score (การสอบเทียบ) | False Negatives (หลุดรอด) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost (Champion)** | **0.9323** | **0.8189** | **93.08%** | **71.29%** | **0.0521** | **408 (ลดเหลือ 292 ด้วย Cost-Optimal)** |
| 🌲 **Random Forest** | 0.9281 | 0.8118 | 92.68% | 71.92% | 0.0553 | 399 |
| 📐 **Logistic Regression** | 0.8623 | 0.6344 | 80.83% | 74.74% | 0.1342 | 359 |

---

## 🧪 การทดสอบระบบอัตโนมัติ (Automated Unit Testing)

รันชุดทดสอบเพื่อยืนยันความถูกต้องของ Pipeline และ XAI:
```bash
python -m pytest tests/test_pipeline.py -v
```
*(ผ่านการทดสอบ 5/5 ข้อ ครอบคลุม: การตัดเกรดความเสี่ยง, การแปลงคุณลักษณะ, การบันทึกฐานข้อมูล, การจำลอง Stress Testing, และการประเมินความเป็นธรรมของโมเดล)*

---

## 📂 โครงสร้างโฟลเดอร์โครงการ (Project Structure)

```
credit_risk/
├── data/
│   └── credit_risk_dataset.csv     # ข้อมูลสินเชื่อหลัก 32,581 รายการ (Kaggle Dataset)
├── models/
│   ├── best_model.joblib           # โมเดล XGBoost Champion ที่ผ่านการปรับจูน
│   ├── logreg_model.joblib         # โมเดล Logistic Regression
│   ├── rf_model.joblib             # โมเดล Random Forest
│   ├── preprocessor.joblib         # ColumnTransformer และ Advanced Feature Engineering
│   ├── shap_explainer.joblib       # SHAP TreeExplainer
│   ├── evaluation_results.json     # สรุปผล Performance Metrics และ Cost Analysis
│   └── sample_test_cases.csv       # ตัวอย่างเคสทดสอบสำหรับโหลดในหน้าเว็บ
├── pages/
│   ├── 1_📉_Macro_Stress_Testing.py # พอร์ทัลทดสอบภาวะวิกฤตเศรษฐกิจ (Basel III)
│   ├── 2_⚖️_AI_Fairness_&_Ethics.py # พอร์ทัลตรวจสอบจริยธรรมและความเป็นธรรมของ AI
│   └── 3_📡_API_Playground.py      # หน้าต่างทดสอบการเชื่อมต่อ REST API
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # การคลีนข้อมูล, Outlier Treatment, และคำนวณ Feature Ratios
│   ├── train.py                    # ไปป์ไลน์เทรน, SMOTE, และ Cost Threshold Optimization
│   ├── explainability.py           # การคำนวณ SHAP Waterfall พร้อมแปลความหมายภาษาไทย
│   ├── counterfactual.py           # ระบบจำลองเงื่อนไขสินเชื่อ (What-If / Recourse Engine)
│   ├── database.py                 # ระบบฐานข้อมูล SQLite และ Audit Logging
│   ├── pdf_generator.py            # ระบบสร้างเอกสารรับรองสินเชื่อ PDF ฉบับทางการ (ReportLab)
│   ├── stress_testing.py           # โมดูลคำนวณ Macro Stress Testing
│   ├── fairness.py                 # โมดูลคำนวณ Four-Fifths Rule Disparate Impact
│   └── utils.py                    # ตารางเกณฑ์ความเสี่ยง (0-100) และ Persona ข้อมูลตัวอย่าง
├── tests/
│   └── test_pipeline.py            # ชุดทดสอบอัตโนมัติ Unit Tests (5/5 passed)
├── app.py                          # เว็บแอปพลิเคชัน Streamlit Dashboard ระดับสถาบันการเงิน
├── api.py                          # บริการ REST API Backend ด้วย FastAPI
├── setup.bat                       # สคริปต์ติดตั้งระบบอัตโนมัติ 1-Click บน Windows
├── run_app.bat                     # สคริปต์รันหน้าเว็บ Streamlit Dashboard บน Windows
├── run_api.bat                     # สคริปต์รัน FastAPI Backend บน Windows
├── requirements.txt                # รายการไลบรารี Python ที่จำเป็นทั้งหมด
├── PROJECT_MASTER_PROMPT.md        # พร้อมพ์หลักฉบับสมบูรณ์สำหรับ AI LLM
├── .gitignore                      # กฎยกเว้นไฟล์ที่ไม่จำเป็นในการขึ้น Git
└── README.md                       # เอกสารคู่มือโครงการฉบับสมบูรณ์
```

---

## 📜 ลิขสิทธิ์และการอ้างอิง (Citation & Academic Acknowledgement)
โครงงานนี้พัฒนาขึ้นเพื่อประกอบการเรียนวิชา 240-403 การเตรียมความพร้อมสำหรับสหกิจศึกษา มหาวิทยาลัยสงขลานครินทร์ วิทยาเขตหาดใหญ่ ตามมาตรฐาน AI Governance และ Responsible AI
