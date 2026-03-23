# 🏷️ Mercari Price Suggestion Web App

แอปพลิเคชัน AI สำหรับประเมินและแนะนำราคาสินค้ามือสองอ้างอิงจากฐานข้อมูลของ Mercari พัฒนาด้วย Machine Learning และแสดงผลผ่าน Streamlit

## 📌 เกี่ยวกับโปรเจค (About the Project)
โปรเจคนี้เป็นการนำโมเดล Machine Learning มาแก้ปัญหาการตั้งราคาสินค้าออนไลน์ โดย AI จะเรียนรู้จากข้อมูลสินค้าในอดีต (ชื่อ, แบรนด์, หมวดหมู่, สภาพ, ค่าจัดส่ง และ รายละเอียดสินค้า) เพื่อทำนายราคาที่เหมาะสมที่สุดให้กับผู้ขาย

**โมเดลที่ใช้:** `Ridge Regression` 
* เหตุผลที่เลือกใช้: เนื่องจากข้อมูลมีการแปลงข้อความ (Text) ด้วยเทคนิค TF-IDF ทำให้เกิดฟีเจอร์จำนวนมาก โมเดล Ridge ซึ่งมี Regularization ช่วยควบคุม Overfitting ได้ดีและประมวลผลได้อย่างรวดเร็ว

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)
* **ภาษา:** Python 3
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn (Ridge Regression, TF-IDF, One-Hot Encoding)
* **Data Manipulation:** Pandas, NumPy

## 🚀 วิธีการติดตั้งและทดลองรัน (How to Run Locally)
หากต้องการนำโปรเจคนี้ไปรันบนเครื่องคอมพิวเตอร์ของคุณ สามารถทำตามขั้นตอนได้ดังนี้:

1. Clone repository นี้ลงเครื่องของคุณ
2. ติดตั้งไลบรารีที่จำเป็นผ่านไฟล์ `requirements.txt`
   ```bash
   pip install -r requirements.txt