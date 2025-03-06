import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import datetime
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
import streamlit.components.v1 as components

# ตั้งค่าฟอนต์ให้รองรับภาษาไทย
matplotlib.rc("font", family="Tahoma")

# ตั้งค่าธีมสีที่ทันสมัย (พื้นหลังสีเข้ม)
st.set_page_config(
    page_title="ระบบวิเคราะห์โรคและการขาดธาตุอาหารจากใบของพืช", 
    layout="wide",
    page_icon="images/durian_leaf1.png")
# แสดงโลโก้ในหน้าแอป
st.image("images/durian_leaf1.png", width=100)  # ปรับขนาดตามต้องการ
st.title("🌿 ระบบวิเคราะห์โรคและธาตุอาหารใบพืช")
st.write("แอปนี้ใช้สำหรับวิเคราะห์ใบพืช")

st.markdown("""
<style>
    .main {background-color: #121212; padding: 20px;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 12px; padding: 10px 20px; font-weight: bold;}
    .stProgress .st-bo {background-color: #4CAF50;}
    .highlight-box {background-color: #1e1e1e; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(255,255,255,0.1); margin-bottom: 20px;}
    .metric-card {background-color: #262626; padding: 15px; border-radius: 12px; box-shadow: 0 2px 6px rgba(255,255,255,0.1); text-align: center; color: #ffffff; margin: 10px 0;}
    .card {background-color: #1e1e1e; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(255,255,255,0.1); margin-bottom: 20px;}
    h1, h2, h3, p, div {color: #e0e0e0; font-family: 'Arial', sans-serif;}
    .centered-image {display: flex; justify-content: center; align-items: center; margin: 20px 0;}
    .stTabs [data-baseweb="tab-list"] {background-color: #1e1e1e; border-radius: 10px; padding: 5px;}
    .stTabs [data-baseweb="tab"] {color: #e0e0e0; border-radius: 8px; padding: 10px 20px; margin: 0 5px;}
    .stTabs [aria-selected="true"] {background-color: #4CAF50; color: white;}
    .icon {vertical-align: middle; margin-right: 8px;}
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันสำหรับสร้างการ์ดแบบกำหนดเองด้วย HTML/CSS
def create_card(title, content, icon="📊"):
    components.html(f"""
    <div class="card">
        <h3><span class="icon">{icon}</span>{title}</h3>
        <p>{content}</p>
    </div>
    """, height=150)

# ฟังก์ชันวิเคราะห์พื้นผิวด้วย GLCM
def texture_analysis(image_rgb):
    img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation
    }

# ฟังก์ชันวิเคราะห์ใบพืช
def analyze_leaf(image):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    kernel = np.ones((5, 5), np.uint8)
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    brown_mask = cv2.inRange(img_hsv, lower_brown, upper_brown)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_purple = np.array([120, 40, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(img_hsv, lower_purple, upper_purple)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_edge_brown = np.array([5, 50, 20])
    upper_edge_brown = np.array([20, 255, 100])
    edge_brown_mask = cv2.inRange(img_hsv, lower_edge_brown, upper_edge_brown)
    edge_brown_mask = cv2.morphologyEx(edge_brown_mask, cv2.MORPH_OPEN, kernel)
    edge_brown_mask = cv2.morphologyEx(edge_brown_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_gray = np.array([0, 0, 150])
    upper_gray = np.array([180, 30, 255])
    gray_mask = cv2.inRange(img_hsv, lower_gray, upper_gray)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_pale_yellow = np.array([25, 20, 150])
    upper_pale_yellow = np.array([35, 100, 255])
    pale_yellow_mask = cv2.inRange(img_hsv, lower_pale_yellow, upper_pale_yellow)
    pale_yellow_mask = cv2.morphologyEx(pale_yellow_mask, cv2.MORPH_OPEN, kernel)
    pale_yellow_mask = cv2.morphologyEx(pale_yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    lower_dark_brown = np.array([5, 50, 20])
    upper_dark_brown = np.array([20, 255, 100])
    dark_brown_mask = cv2.inRange(img_hsv, lower_dark_brown, upper_dark_brown)
    dark_brown_mask = cv2.morphologyEx(dark_brown_mask, cv2.MORPH_OPEN, kernel)
    dark_brown_mask = cv2.morphologyEx(dark_brown_mask, cv2.MORPH_CLOSE, kernel)
    
    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
    brown_pct = float((np.sum(brown_mask > 0) / total_pixels) * 100)
    yellow_pct = float((np.sum(yellow_mask > 0) / total_pixels) * 100)
    purple_pct = float((np.sum(purple_mask > 0) / total_pixels) * 100)
    edge_brown_pct = float((np.sum(edge_brown_mask > 0) / total_pixels) * 100)
    gray_pct = float((np.sum(gray_mask > 0) / total_pixels) * 100)
    pale_yellow_pct = float((np.sum(pale_yellow_mask > 0) / total_pixels) * 100)
    dark_brown_pct = float((np.sum(dark_brown_mask > 0) / total_pixels) * 100)
    
    disease_prob = float((brown_pct + yellow_pct + purple_pct + edge_brown_pct + gray_pct + pale_yellow_pct + dark_brown_pct) / 7)
    
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
    
    texture_features = texture_analysis(img_rgb)
    
    return (brown_highlight, yellow_highlight, purple_highlight, edge_brown_highlight, gray_highlight, 
            pale_yellow_highlight, dark_brown_highlight, img_rgb, brown_pct, yellow_pct, purple_pct, 
            edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, disease_prob, ph_status, texture_features)

# ฟังก์ชันกราฟ RGB
def plot_rgb_histogram(image):
    r, g, b = cv2.split(image)
    plt.figure(figsize=(10, 4), facecolor="#121212")
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

# ฟังก์ชันวินิจฉัย
def disease_diagnosis(brown_pct, yellow_pct, purple_pct, edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, texture_features):
    diagnoses = {
        "โรค": [],
        "ธาตุอาหารหลัก (N-P-K)": [],
        "ธาตุอาหารรอง (Ca-Mg-S)": [],
        "จุลธาตุ": []
    }
    
    contrast = texture_features["contrast"]
    homogeneity = texture_features["homogeneity"]
    energy = texture_features["energy"]
    correlation = texture_features["correlation"]
    
    if brown_pct > 5.0:
        if contrast > 150 and homogeneity < 0.5:
            diagnoses["โรค"].append({
                "name": "โรคใบไหม้",
                "description": "ใบมีจุดสีน้ำตาลจากการติดเชื้อรา พื้นผิวหยาบ",
                "recommendation": "ใช้สารฆ่าเชื้อราและตัดใบที่ติดเชื้อออก"
            })
        else:
            diagnoses["จุลธาตุ"].append({
                "name": "ขาดทองแดง (Cu)",
                "description": "ใบน้ำตาล พื้นผิวเรียบ",
                "recommendation": "ใส่ทองแดงซัลเฟต"
            })
    
    if gray_pct > 3.0:
        if contrast > 100 and homogeneity < 0.6:
            diagnoses["โรค"].append({
                "name": "โรคราแป้ง",
                "description": "ใบมีฝ้าสีขาวหรือเทาจากเชื้อรา พื้นผิวหยาบ",
                "recommendation": "ใช้สารกำจัดเชื้อราและปรับความชื้น"
            })
        else:
            diagnoses["ธาตุอาหารรอง (Ca-Mg-S)"].append({
                "name": "ขาดแคลเซียม (Ca)",
                "description": "ใบซีดหรือมีจุดขาว พื้นผิวเรียบ",
                "recommendation": "ใส่ปูนขาวหรือแคลเซียมไนเตรต"
            })
    
    if dark_brown_pct > 4.0:
        if energy < 0.2:
            diagnoses["โรค"].append({
                "name": "โรคจุดใบ/เน่า",
                "description": "ใบมีจุดสีน้ำตาลเข้มถึงดำ พื้นผิวไม่สม่ำเสมอ",
                "recommendation": "กำจัดใบที่ติดเชื้อและใช้สารฆ่าเชื้อ"
            })
        else:
            diagnoses["จุลธาตุ"].append({
                "name": "ขาดสังกะสี (Zn)",
                "description": "ใบมีจุดน้ำตาลเข้ม พื้นผิวสม่ำเสมอ",
                "recommendation": "ใส่สังกะสีซัลเฟต"
            })
    
    if yellow_pct > 10.0:
        diagnoses["ธาตุอาหารหลัก (N-P-K)"].append({
            "name": "ขาดไนโตรเจน (N)",
            "description": "ใบเหลืองทั่วทั้งใบ",
            "recommendation": "ใส่ปุ๋ยไนโตรเจน เช่น ยูเรีย"
        })
    if purple_pct > 5.0:
        diagnoses["ธาตุอาหารหลัก (N-P-K)"].append({
            "name": "ขาดฟอสฟอรัส (P)",
            "description": "ใบมีสีม่วงหรือแดง",
            "recommendation": "ใส่ปุ๋ยฟอสเฟต"
        })
    if edge_brown_pct > 4.0:
        diagnoses["ธาตุอาหารหลัก (N-P-K)"].append({
            "name": "ขาดโพแทสเซียม (K)",
            "description": "ขอบใบไหม้สีน้ำตาล",
            "recommendation": "ใส่ปุ๋ยโพแทสเซียม เช่น โพแทสเซียมคลอไรด์"
        })
    
    if pale_yellow_pct > 5.0:
        diagnoses["ธาตุอาหารรอง (Ca-Mg-S)"].append({
            "name": "ขาดแมกนีเซียม (Mg)",
            "description": "ใบเหลืองอ่อนระหว่างเส้นใบ",
            "recommendation": "ใส่แมกนีเซียมซัลเฟต"
        })
        diagnoses["จุลธาตุ"].append({
            "name": "ขาดเหล็ก (Fe)",
            "description": "ใบซีดเหลืองระหว่างเส้นใบ",
            "recommendation": "ใส่ธาตุเหล็กคีเลต"
        })
    
    return diagnoses

# ฟังก์ชันบันทึกข้อมูล
def save_analysis_results(timestamp, disease_prob, ph_status, diagnoses, texture_features):
    data = {
        "Timestamp": [timestamp],
        "Disease_Probability": [disease_prob],
        "PH_Status": [ph_status],
        "Diagnoses": [str(diagnoses)],
        "Contrast": [texture_features["contrast"]],
        "Homogeneity": [texture_features["homogeneity"]],
        "Energy": [texture_features["energy"]],
        "Correlation": [texture_features["correlation"]]
    }
    df = pd.DataFrame(data)
    
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")
    
    file_path = f"analysis_results/analysis_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    st.success(f"บันทึกผลการวิเคราะห์ในไฟล์: {file_path}")

# ฟังก์ชันประมวลผลภาพจากกล้อง
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.camera_facing = "environment"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if st.session_state.get("capture", False):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"captured_image_{timestamp}.jpg", img)
            st.session_state["capture"] = False
            st.session_state["captured_image"] = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ส่วนหัวของแอป
st.title("🌿 ระบบวิเคราะห์โรคและการขาดธาตุอาหารจากใบของพืช")
st.markdown("หน่วยวิจัยปัญญาประดิษฐ์และการเรียนรู้เชิงลึกเพื่อการเกษตรดิจิทัล", unsafe_allow_html=True)

# แท็บหลัก: อัปโหลดภาพ vs เปิดกล้อง
input_tab1, input_tab2 = st.tabs(["📸 อัปโหลดภาพ", "📷 เปิดกล้อง"])

# แท็บอัปโหลดภาพ
with input_tab1:
    st.subheader("อัปโหลดภาพใบพืช")
    uploaded_file = st.file_uploader("เลือกภาพใบพืช (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], 
                                   help="รองรับไฟล์ภาพขนาดไม่เกิน 5MB")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_resized = image.resize((500, int(500 * image.height / image.width)))
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(image_resized, caption="ภาพใบพืชที่อัปโหลด", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner("กำลังวิเคราะห์ภาพ..."):
            (brown_highlight, yellow_highlight, purple_highlight, edge_brown_highlight, gray_highlight, 
             pale_yellow_highlight, dark_brown_highlight, img_rgb, brown_pct, yellow_pct, purple_pct, 
             edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, disease_prob, ph_status, texture_features) = analyze_leaf(image)
        
        # แท็บย่อยสำหรับผลการวิเคราะห์
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["📊 ผลการวิเคราะห์สี", "🖼️ การวิเคราะห์พื้นผิว", "⚗️ ค่า pH", "🏥 คำแนะนำ"])
        
        with result_tab1:
            st.subheader("ผลการวิเคราะห์สี")
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                brown_resized = Image.fromarray(brown_highlight).resize((200, int(200 * brown_highlight.shape[0] / brown_highlight.shape[1])))
                st.image(brown_resized, caption="สีน้ำตาล (โรคใบไหม้)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดทองแดง: <b>{brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                purple_resized = Image.fromarray(purple_highlight).resize((200, int(200 * purple_highlight.shape[0] / purple_highlight.shape[1])))
                st.image(purple_resized, caption="สีม่วง (ขาดฟอสฟอรัส)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดฟอสฟอรัส: <b>{purple_pct:.2f}%</b></div>', unsafe_allow_html=True)
                gray_resized = Image.fromarray(gray_highlight).resize((200, int(200 * gray_highlight.shape[0] / gray_highlight.shape[1])))
                st.image(gray_resized, caption="สีเทา/ขาว (โรคราแป้ง/ขาดแคลเซียม)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดแคลเซียม: <b>{gray_pct:.2f}%</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                yellow_resized = Image.fromarray(yellow_highlight).resize((200, int(200 * yellow_highlight.shape[0] / yellow_highlight.shape[1])))
                st.image(yellow_resized, caption="สีเหลือง (ขาดไนโตรเจน)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดไนโตรเจน: <b>{yellow_pct:.2f}%</b></div>', unsafe_allow_html=True)
                edge_brown_resized = Image.fromarray(edge_brown_highlight).resize((200, int(200 * edge_brown_highlight.shape[0] / edge_brown_highlight.shape[1])))
                st.image(edge_brown_resized, caption="ขอบน้ำตาล (ขาดโพแทสเซียม)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดโพแทสเซียม: <b>{edge_brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                pale_yellow_resized = Image.fromarray(pale_yellow_highlight).resize((200, int(200 * pale_yellow_highlight.shape[0] / pale_yellow_highlight.shape[1])))
                st.image(pale_yellow_resized, caption="สีเหลืองอ่อน (ขาดแมกนีเซียม/เหล็ก)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดแมกนีเซียม: <b>{pale_yellow_pct:.2f}%</b></div>', unsafe_allow_html=True)
                dark_brown_resized = Image.fromarray(dark_brown_highlight).resize((200, int(200 * dark_brown_highlight.shape[0] / dark_brown_highlight.shape[1])))
                st.image(dark_brown_resized, caption="สีน้ำตาลเข้ม (จุดใบ/ขาดสังกะสี)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดสังกะสี: <b>{dark_brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # กราฟ RGB
            st.subheader("การกระจายตัวของค่าสี RGB")
            plot_rgb_histogram(img_rgb)
            
            # ความน่าจะเป็นปัญหา
            st.subheader("ความน่าจะเป็นของปัญหา")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.progress(min(disease_prob / 100, 1.0))
            create_card("โอกาสเกิดปัญหา", f"ใบนี้มีโอกาสเกิดปัญหา: <b>{disease_prob:.2f}%</b>", icon="⚠️")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab2:
            st.subheader("การวิเคราะห์พื้นผิว")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            create_card("Contrast", f"ความแตกต่างของความเข้ม: <b>{texture_features['contrast']:.2f}</b>", icon="🎨")
            create_card("Homogeneity", f"ความสม่ำเสมอ: <b>{texture_features['homogeneity']:.2f}</b>", icon="🖌️")
            create_card("Energy", f"พลังงาน: <b>{texture_features['energy']:.2f}</b>", icon="⚡")
            create_card("Correlation", f"ความสัมพันธ์: <b>{texture_features['correlation']:.2f}</b>", icon="🔗")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab3:
            st.subheader("การประเมินค่า pH")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            create_card("สถานะ pH", f"<b>{ph_status}</b>", icon="⚗️")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab4:
            st.subheader("การวินิจฉัยและคำแนะนำ")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            diagnoses = disease_diagnosis(brown_pct, yellow_pct, purple_pct, edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, texture_features)
            
            for category, issues in diagnoses.items():
                if issues:
                    with st.expander(f"📌 {category}", expanded=True):
                        for issue in issues:
                            create_card(f"{issue['name']}", f"ลักษณะอาการ: {issue['description']}<br>คำแนะนำ: {issue['recommendation']}")
            if not any(diagnoses.values()):
                st.success("✅ ไม่พบโรคหรือการขาดธาตุอาหารที่ชัดเจน")
            st.markdown('</div>', unsafe_allow_html=True)

        # ปุ่มบันทึกผล
        if st.button("💾 บันทึกผลการวิเคราะห์"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_analysis_results(timestamp, disease_prob, ph_status, diagnoses, texture_features)

# แท็บเปิดกล้อง
with input_tab2:
    st.subheader("เปิดกล้องสมาร์ทโฟน")
    camera_options = st.radio("เลือกกล้อง", ["กล้องหลัง", "กล้องหน้า"], horizontal=True)
    
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    if "capture" not in st.session_state:
        st.session_state["capture"] = False
    if "captured_image" not in st.session_state:
        st.session_state["captured_image"] = None

    class CameraProcessor(VideoProcessorBase):
        def __init__(self):
            self.camera_facing = "environment" if camera_options == "กล้องหลัง" else "user"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if st.session_state.get("capture", False):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"captured_image_{timestamp}.jpg", img)
                st.session_state["capture"] = False
                st.session_state["captured_image"] = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CameraProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

    if st.button("📸 ถ่ายภาพ"):
        st.session_state["capture"] = True

    if st.session_state.get("captured_image") is not None:
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(st.session_state["captured_image"], caption="ภาพที่ถ่ายจากกล้อง", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner("กำลังวิเคราะห์ภาพ..."):
            (brown_highlight, yellow_highlight, purple_highlight, edge_brown_highlight, gray_highlight, 
             pale_yellow_highlight, dark_brown_highlight, img_rgb, brown_pct, yellow_pct, purple_pct, 
             edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, disease_prob, ph_status, texture_features) = analyze_leaf(st.session_state["captured_image"])
        
        # แท็บย่อยสำหรับผลการวิเคราะห์
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["📊 ผลการวิเคราะห์สี", "🖼️ การวิเคราะห์พื้นผิว", "⚗️ ค่า pH", "🏥 คำแนะนำ"])
        
        with result_tab1:
            st.subheader("ผลการวิเคราะห์สี")
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                brown_resized = Image.fromarray(brown_highlight).resize((200, int(200 * brown_highlight.shape[0] / brown_highlight.shape[1])))
                st.image(brown_resized, caption="สีน้ำตาล (โรคใบไหม้)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดทองแดง: <b>{brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                purple_resized = Image.fromarray(purple_highlight).resize((200, int(200 * purple_highlight.shape[0] / purple_highlight.shape[1])))
                st.image(purple_resized, caption="สีม่วง (ขาดฟอสฟอรัส)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดฟอสฟอรัส: <b>{purple_pct:.2f}%</b></div>', unsafe_allow_html=True)
                gray_resized = Image.fromarray(gray_highlight).resize((200, int(200 * gray_highlight.shape[0] / gray_highlight.shape[1])))
                st.image(gray_resized, caption="สีเทา/ขาว (โรคราแป้ง/ขาดแคลเซียม)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดแคลเซียม: <b>{gray_pct:.2f}%</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                yellow_resized = Image.fromarray(yellow_highlight).resize((200, int(200 * yellow_highlight.shape[0] / yellow_highlight.shape[1])))
                st.image(yellow_resized, caption="สีเหลือง (ขาดไนโตรเจน)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดไนโตรเจน: <b>{yellow_pct:.2f}%</b></div>', unsafe_allow_html=True)
                edge_brown_resized = Image.fromarray(edge_brown_highlight).resize((200, int(200 * edge_brown_highlight.shape[0] / edge_brown_highlight.shape[1])))
                st.image(edge_brown_resized, caption="ขอบน้ำตาล (ขาดโพแทสเซียม)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดโพแทสเซียม: <b>{edge_brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                pale_yellow_resized = Image.fromarray(pale_yellow_highlight).resize((200, int(200 * pale_yellow_highlight.shape[0] / pale_yellow_highlight.shape[1])))
                st.image(pale_yellow_resized, caption="สีเหลืองอ่อน (ขาดแมกนีเซียม/เหล็ก)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดแมกนีเซียม: <b>{pale_yellow_pct:.2f}%</b></div>', unsafe_allow_html=True)
                dark_brown_resized = Image.fromarray(dark_brown_highlight).resize((200, int(200 * dark_brown_highlight.shape[0] / dark_brown_highlight.shape[1])))
                st.image(dark_brown_resized, caption="สีน้ำตาลเข้ม (จุดใบ/ขาดสังกะสี)", use_container_width=True)
                st.markdown(f'<div class="metric-card">โอกาสขาดสังกะสี: <b>{dark_brown_pct:.2f}%</b></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("การกระจายตัวของค่าสี RGB")
            plot_rgb_histogram(img_rgb)
            
            st.subheader("ความน่าจะเป็นของปัญหา")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.progress(min(disease_prob / 100, 1.0))
            create_card("โอกาสเกิดปัญหา", f"ใบนี้มีโอกาสเกิดปัญหา: <b>{disease_prob:.2f}%</b>", icon="⚠️")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab2:
            st.subheader("การวิเคราะห์พื้นผิว")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            create_card("Contrast", f"ความแตกต่างของความเข้ม: <b>{texture_features['contrast']:.2f}</b>", icon="🎨")
            create_card("Homogeneity", f"ความสม่ำเสมอ: <b>{texture_features['homogeneity']:.2f}</b>", icon="🖌️")
            create_card("Energy", f"พลังงาน: <b>{texture_features['energy']:.2f}</b>", icon="⚡")
            create_card("Correlation", f"ความสัมพันธ์: <b>{texture_features['correlation']:.2f}</b>", icon="🔗")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab3:
            st.subheader("การประเมินค่า pH")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            create_card("สถานะ pH", f"<b>{ph_status}</b>", icon="⚗️")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_tab4:
            st.subheader("การวินิจฉัยและคำแนะนำ")
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            diagnoses = disease_diagnosis(brown_pct, yellow_pct, purple_pct, edge_brown_pct, gray_pct, pale_yellow_pct, dark_brown_pct, texture_features)
            
            for category, issues in diagnoses.items():
                if issues:
                    with st.expander(f"📌 {category}", expanded=True):
                        for issue in issues:
                            create_card(f"{issue['name']}", f"ลักษณะอาการ: {issue['description']}<br>คำแนะนำ: {issue['recommendation']}")
            if not any(diagnoses.values()):
                st.success("✅ ไม่พบโรคหรือการขาดธาตุอาหารที่ชัดเจน")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("💾 บันทึกผลการวิเคราะห์"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_analysis_results(timestamp, disease_prob, ph_status, diagnoses, texture_features)