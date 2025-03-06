import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# ตั้งค่าฟอนต์ให้รองรับภาษาไทย
matplotlib.rc("font", family="Tahoma")

# ตั้งค่าธีมสีที่ทันสมัย (พื้นหลังสีดำ) และเปลี่ยนไอคอน
st.set_page_config(
    page_title="ระบบวิเคราะห์โรคและธาตุอาหารใบพืช",
    layout="wide",
    page_icon="durian_leaf1.png"  # เปลี่ยนเป็น path ของไฟล์โลโก้คุณ
)

st.markdown("""
<style>
    .main {background-color: #000000;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    .stProgress .st-bo {background-color: #4CAF50;}
    .highlight-box {background-color: #1a1a1a; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(255,255,255,0.1);}
    .metric-card {background-color: #1a1a1a; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(255,255,255,0.1); text-align: center; color: #ffffff;}
    h1, h2, h3, p, div {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# แสดงโลโก้ข้างชื่อ
st.image("durian_leaf1.png", width=50)  # เปลี่ยนเป็น path ของไฟล์โลโก้คุณ
st.title("🌿 WEB APP วิเคราะห์โรคและธาตุอาหารใบพืช")

def analyze_leaf(image):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    brown_mask = cv2.inRange(img_hsv, lower_brown, upper_brown)
    
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    
    lower_purple = np.array([120, 40, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(img_hsv, lower_purple, upper_purple)
    
    lower_edge_brown = np.array([5, 50, 20])
    upper_edge_brown = np.array([20, 255, 100])
    edge_brown_mask = cv2.inRange(img_hsv, lower_edge_brown, upper_edge_brown)
    
    lower_gray = np.array([0, 0, 150])
    upper_gray = np.array([180, 30, 255])
    gray_mask = cv2.inRange(img_hsv, lower_gray, upper_gray)
    
    lower_pale_yellow = np.array([25, 20, 150])
    upper_pale_yellow = np.array([35, 100, 255])
    pale_yellow_mask = cv2.inRange(img_hsv, lower_pale_yellow, upper_pale_yellow)
    
    lower_dark_brown = np.array([5, 50, 20])
    upper_dark_brown = np.array([20, 255, 100])
    dark_brown_mask = cv2.inRange(img_hsv, lower_dark_brown, upper_dark_brown)
    
    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
    brown_pct = (np.sum(brown_mask > 0) / total_pixels) * 100
    yellow_pct = (np.sum(yellow_mask > 0) / total_pixels) * 100
    purple_pct = (np.sum(purple_mask > 0) / total_pixels) * 100
    edge_brown_pct = (np.sum(edge_brown_mask > 0) / total_pixels) * 100
    gray_pct = (np.sum(gray_mask > 0) / total_pixels) * 100
    pale_yellow_pct = (np.sum(pale_yellow_mask > 0) / total_pixels) * 100
    dark_brown_pct = (np.sum(dark_brown_mask > 0) / total_pixels) * 100
    
    disease_prob = (brown_pct + yellow_pct + purple_pct + edge_brown_pct + gray_pct + pale_yellow_pct + dark_brown_pct) / 7
    
    if yellow_pct > 10 or pale_yellow_pct > 10:
        ph_status = "ดินอาจเป็นกรดสูง (pH < 6)"
    elif purple_pct > 5:
        ph_status = "ดินอาจเป็นด่างสูง (pH > 7.5)"
    else:
        ph_status = "ค่า pH อยู่ในช่วงปกติ (6-7.5)"
    
    brown_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=brown_mask)
    yellow_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
    purple_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=purple_mask)
    edge_brown_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=edge_brown_mask)
    gray_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=gray_mask)
    pale_yellow_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=pale_yellow_mask)
    dark_brown_highlight = cv2.bitwise_and(img_rgb, img_rgb, mask=dark_brown_mask)
    
    return (brown_highlight, yellow_highlight, purple_highlight, edge_brown_highlight, gray_highlight, 
            pale_yellow_highlight, dark_brown_highlight, img_rgb, brown_pct, yellow_pct, purple_pct, 
            edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, disease_prob, ph_status)

def plot_rgb_histogram(image):
    r, g, b = cv2.split(image)
    plt.figure(figsize=(10, 4), facecolor="#000000")
    plt.hist(r.ravel(), bins=256, color='#FF6B6B', alpha=0.7, label='แดง')
    plt.hist(g.ravel(), bins=256, color='#4CAF50', alpha=0.7, label='เขียว')
    plt.hist(b.ravel(), bins=256, color='#4D96FF', alpha=0.7, label='น้ำเงิน')
    plt.title("การกระจายตัวของค่าสี RGB", fontsize=14, pad=10, color='white')
    plt.xlabel("ค่าพิกเซล", fontsize=12, color='white')
    plt.ylabel("ความถี่", fontsize=12, color='white')
    plt.legend(prop={'size': 10}, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)

def disease_diagnosis(brown_pct, yellow_pct, purple_pct, edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct):
    diagnoses = {
        "โรค": [],
        "ธาตุอาหารหลัก (NPK)": [],
        "ธาตุอาหารรอง": [],
        "จุลธาตุ": []
    }
    
    if brown_pct > 5.0:
        diagnoses["โรค"].append({"name": "โรคใบไหม้", "description": "ใบมีจุดสีน้ำตาลจากการติดเชื้อรา", "recommendation": "ใช้สารฆ่าเชื้อราและตัดใบที่ติดเชื้อออก"})
    if gray_pct > 3.0:
        diagnoses["โรค"].append({"name": "โรคราแป้ง", "description": "ใบมีฝ้าสีขาวหรือเทาจากเชื้อรา", "recommendation": "ใช้สารกำจัดเชื้อราและปรับความชื้น"})
    if dark_brown_pct > 4.0:
        diagnoses["โรค"].append({"name": "โรคจุดใบ/เน่า", "description": "ใบมีจุดสีน้ำตาลเข้มถึงดำ", "recommendation": "กำจัดใบที่ติดเชื้อและใช้สารฆ่าเชื้อ"})
    
    if yellow_pct > 10.0:
        diagnoses["ธาตุอาหารหลัก (NPK)"].append({"name": "ขาดไนโตรเจน (N)", "description": "ใบเหลืองทั่วทั้งใบ", "recommendation": "ใส่ปุ๋ยไนโตรเจน เช่น ยูเรีย"})
    if purple_pct > 5.0:
        diagnoses["ธาตุอาหารหลัก (NPK)"].append({"name": "ขาดฟอสฟอรัส (P)", "description": "ใบมีสีม่วงหรือแดง", "recommendation": "ใส่ปุ๋ยฟอสเฟต"})
    if edge_brown_pct > 4.0:
        diagnoses["ธาตุอาหารหลัก (NPK)"].append({"name": "ขาดโพแทสเซียม (K)", "description": "ขอบใบไหม้สีน้ำตาล", "recommendation": "ใส่ปุ๋ยโพแทสเซียม เช่น โพแทสเซียมคลอไรด์"})
    
    if gray_pct > 3.0 and brown_pct < 5.0:
        diagnoses["ธาตุอาหารรอง"].append({"name": "ขาดแคลเซียม (Ca)", "description": "ใบซีดหรือมีจุดขาว", "recommendation": "ใส่ปูนขาวหรือแคลเซียมไนเตรต"})
    if pale_yellow_pct > 5.0:
        diagnoses["ธาตุอาหารรอง"].append({"name": "ขาดแมกนีเซียม (Mg)", "description": "ใบเหลืองอ่อนระหว่างเส้นใบ", "recommendation": "ใส่แมกนีเซียมซัลเฟต"})
    if dark_brown_pct > 4.0 and yellow_pct < 10.0:
        diagnoses["ธาตุอาหารรอง"].append({"name": "ขาดกำมะถัน (S)", "description": "ใบมีจุดน้ำตาลเข้ม", "recommendation": "ใส่ปุ๋ยที่มีกำมะถัน"})
    
    if pale_yellow_pct > 5.0 and yellow_pct < 10.0:
        diagnoses["จุลธาตุ"].append({"name": "ขาดเหล็ก (Fe)", "description": "ใบซีดเหลืองระหว่างเส้นใบ", "recommendation": "ใส่ธาตุเหล็กคีเลต"})
    if dark_brown_pct > 4.0 and edge_brown_pct < 4.0:
        diagnoses["จุลธาตุ"].append({"name": "ขาดสังกะสี (Zn)", "description": "ใบมีจุดน้ำตาลเข้ม", "recommendation": "ใส่สังกะสีซัลเฟต"})
    if brown_pct > 5.0 and purple_pct > 5.0:
        diagnoses["จุลธาตุ"].append({"name": "ขาดทองแดง (Cu)", "description": "ใบน้ำตาลและม่วง", "recommendation": "ใส่ทองแดงซัลเฟต"})
    
    return diagnoses

# อัปโหลดไฟล์
uploaded_file = st.file_uploader("เลือกภาพใบพืช (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], 
                               help="รองรับไฟล์ภาพขนาดไม่เกิน 5MB")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_resized = image.resize((300, int(300 * image.height / image.width)))
    st.image(image_resized, caption="ภาพใบพืชที่อัปโหลด")
    
    with st.spinner("กำลังวิเคราะห์ภาพ..."):
        (brown_highlight, yellow_highlight, purple_highlight, edge_brown_highlight, gray_highlight, 
         pale_yellow_highlight, dark_brown_highlight, img_rgb, brown_pct, yellow_pct, purple_pct, 
         edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, disease_prob, ph_status) = analyze_leaf(image)
    
    st.subheader("📊 ผลการวิเคราะห์ภาพ")
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        brown_resized = Image.fromarray(brown_highlight).resize((200, int(200 * brown_highlight.shape[0] / brown_highlight.shape[1])))
        st.image(brown_resized, caption="สีน้ำตาล (โรคใบไหม้)")
        st.metric("โอกาสขาดทองแดง", f"{brown_pct:.2f}%", delta=None)
        purple_resized = Image.fromarray(purple_highlight).resize((200, int(200 * purple_highlight.shape[0] / purple_highlight.shape[1])))
        st.image(purple_resized, caption="สีม่วง (ขาดฟอสฟอรัส)")
        st.metric("โอกาสขาดฟอสฟอรัส", f"{purple_pct:.2f}%", delta=None)
        gray_resized = Image.fromarray(gray_highlight).resize((200, int(200 * gray_highlight.shape[0] / gray_highlight.shape[1])))
        st.image(gray_resized, caption="สีเทา/ขาว (โรคราแป้ง/ขาดแคลเซียม)")
        st.metric("โอกาสขาดแคลเซียม", f"{gray_pct:.2f}%", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        yellow_resized = Image.fromarray(yellow_highlight).resize((200, int(200 * yellow_highlight.shape[0] / yellow_highlight.shape[1])))
        st.image(yellow_resized, caption="สีเหลือง (ขาดไนโตรเจน)")
        st.metric("โอกาสขาดไนโตรเจน", f"{yellow_pct:.2f}%", delta=None)
        edge_brown_resized = Image.fromarray(edge_brown_highlight).resize((200, int(200 * edge_brown_highlight.shape[0] / edge_brown_highlight.shape[1])))
        st.image(edge_brown_resized, caption="ขอบน้ำตาล (ขาดโพแทสเซียม)")
        st.metric("โอกาสขาดโพแทสเซียม", f"{edge_brown_pct:.2f}%", delta=None)
        pale_yellow_resized = Image.fromarray(pale_yellow_highlight).resize((200, int(200 * pale_yellow_highlight.shape[0] / pale_yellow_highlight.shape[1])))
        st.image(pale_yellow_resized, caption="สีเหลืองอ่อน (ขาดแมกนีเซียม/เหล็ก)")
        st.metric("โอกาสขาดแมกนีเซียม", f"{pale_yellow_pct:.2f}%", delta=None)
        dark_brown_resized = Image.fromarray(dark_brown_highlight).resize((200, int(200 * dark_brown_highlight.shape[0] / dark_brown_highlight.shape[1])))
        st.image(dark_brown_resized, caption="สีน้ำตาลเข้ม (จุดใบ/ขาดสังกะสี)")
        st.metric("โอกาสขาดสังกะสี", f"{dark_brown_pct:.2f}%", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 การกระจายตัวของค่าสี RGB")
    plot_rgb_histogram(img_rgb)

    st.subheader("🔍 ความน่าจะเป็นของปัญหา")
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.progress(min(disease_prob / 100, 1.0))
    st.write(f"โอกาสที่ใบนี้จะมีปัญหา: **{disease_prob:.2f}%**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("⚗️ การประเมินค่า pH")
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.write(f"สถานะ pH: **{ph_status}**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🏥 การวินิจฉัยและคำแนะนำ")
    diagnoses = disease_diagnosis(brown_pct, yellow_pct, purple_pct, edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct)
    
    for category, issues in diagnoses.items():
        if issues:
            with st.expander(f"📌 {category}", expanded=True):
                for issue in issues:
                    st.markdown(f"**{issue['name']}**")
                    st.write(f"- **ลักษณะอาการ:** {issue['description']}")
                    st.write(f"- **คำแนะนำ:** {issue['recommendation']}")
    if not any(diagnoses.values()):
        st.success("✅ ไม่พบโรคหรือการขาดธาตุอาหารที่ชัดเจน")