# AI Stylist

AI Stylist เป็นแอป Streamlit สำหรับวิเคราะห์โทนสีและสไตล์แฟชั่นจากภาพถ่ายการแต่งตัวของคุณ พร้อมแนะนำโทนสีและแนวสไตล์ที่เหมาะสมโดยอัตโนมัติ

## คุณสมบัติหลัก
- วิเคราะห์ dominant color ของเสื้อและกางเกง
- แสดงโทนสี, HEX, RGB, และชื่อโทน
- แนะนำสีที่เข้ากันตามทฤษฎีสี (Complementary, Analogous, Triadic)
- ทำนายแนวสไตล์แฟชั่นโดย AI (Minimal, Pop, Street, Monochrome, Earth Tone, Pastel, ฯลฯ)
- สรุปแนวสไตล์โดยรวม (Overall Style)
- รองรับการลบพื้นหลังอัตโนมัติ (rembg, backgroundremover, MODNet, heuristic)
- UI สวยงาม รองรับภาษาไทย

## วิธีใช้งาน
1. ติดตั้งไลบรารีที่จำเป็น

```bash
pip install -r requirements.txt
```

2. รันแอป Streamlit

```bash
streamlit run app.py
```

3. เปิดเบราว์เซอร์ที่ลิงก์ที่แสดง (เช่น http://localhost:8501)
4. อัปโหลดภาพถ่ายการแต่งตัว ระบบจะแสดงผลวิเคราะห์สีและสไตล์แฟชั่น

## โครงสร้างไฟล์
- `app.py` : โค้ดหลักของแอป Streamlit
- `color_table_th.py` : ตารางโทนสีและชื่อสี
- `segmentation_utils.py` : ฟังก์ชันแยกส่วนเสื้อ/กางเกง
- `requirements.txt` : รายการไลบรารีที่ต้องติดตั้ง

## ไลบรารีที่ใช้
- streamlit
- numpy
- pillow
- scikit-learn
- webcolors
- rembg
- backgroundremover
- opencv-python
- requests

## เครดิต
- พัฒนาโดย Chanaphon Phetnoi (รหัสนักศึกษา 664230017)
- โค้ดนี้ใช้เพื่อการศึกษาและสาธิตเท่านั้น
