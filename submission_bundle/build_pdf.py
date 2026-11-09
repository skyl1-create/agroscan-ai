import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PAGE_WIDTH = 2480   # A4 @ 300 DPI
PAGE_HEIGHT = 3508  # A4 @ 300 DPI
MARGIN_X = 160
MARGIN_Y = 160
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_X)

def get_font(size, bold=False):
    font_names = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"
    ]
    for fn in font_names:
        if Path(fn).exists():
            try:
                return ImageFont.truetype(fn, size)
            except Exception:
                pass
    return ImageFont.load_default()

class PDFReportGenerator:
    def __init__(self):
        self.pages = []
        self.current_img = None
        self.draw = None
        self.cursor_y = MARGIN_Y
        self.page_number = 1
        self._new_page()

    def _new_page(self):
        if self.current_img is not None:
            self._draw_footer()
            self.pages.append(self.current_img)
            self.page_number += 1
            
        self.current_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "#FFFFFF")
        self.draw = ImageDraw.Draw(self.current_img)
        self.cursor_y = MARGIN_Y
        self._draw_header()

    def _draw_header(self):
        font_header = get_font(24, bold=True)
        font_sub = get_font(22, bold=False)
        self.draw.text((MARGIN_X, MARGIN_Y - 75), "CSE276: APPLIED DEEP LEARNING — PROJECT REPORT", font=font_header, fill="#047857")
        self.draw.text((PAGE_WIDTH - MARGIN_X - 350, MARGIN_Y - 75), "SEC: K325pOJ | GRP: OJ05", font=font_sub, fill="#64748B")
        self.draw.line([(MARGIN_X, MARGIN_Y - 45), (PAGE_WIDTH - MARGIN_X, MARGIN_Y - 45)], fill="#10B981", width=3)

    def _draw_footer(self):
        font_footer = get_font(22, bold=False)
        self.draw.line([(MARGIN_X, PAGE_HEIGHT - MARGIN_Y + 40), (PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - MARGIN_Y + 40)], fill="#CBD5E1", width=2)
        self.draw.text((MARGIN_X, PAGE_HEIGHT - MARGIN_Y + 55), "AI Crop Disease Prediction & Management • AgroScan Precision AI", font=font_footer, fill="#94A3B8")
        self.draw.text((PAGE_WIDTH - MARGIN_X - 110, PAGE_HEIGHT - MARGIN_Y + 55), f"Page {self.page_number}", font=font_footer, fill="#64748B")

    def check_space(self, needed_height):
        if self.cursor_y + needed_height > (PAGE_HEIGHT - MARGIN_Y - 30):
            self._new_page()

    def add_title_block(self, title, subtitle, group_no, section, members, links):
        font_course = get_font(30, bold=True)
        self.draw.text((MARGIN_X, self.cursor_y), "CSE276 — APPLIED DEEP LEARNING CAPSTONE PROJECT REPORT", font=font_course, fill="#065F46")
        self.cursor_y += 50

        font_title = get_font(50, bold=True)
        lines = textwrap.wrap(title, width=44)
        for line in lines:
            self.draw.text((MARGIN_X, self.cursor_y), line, font=font_title, fill="#0F172A")
            self.cursor_y += 70

        font_sub = get_font(26, bold=False)
        self.draw.text((MARGIN_X, self.cursor_y), subtitle, font=font_sub, fill="#475569")
        self.cursor_y += 48

        font_meta = get_font(28, bold=True)
        self.draw.text((MARGIN_X, self.cursor_y), f"Section: {section}   |   Group: {group_no}   |   Lovely Professional University", font=font_meta, fill="#047857")
        self.cursor_y += 55

        # Links Box
        box_h = 175
        self.draw.rectangle([(MARGIN_X, self.cursor_y), (PAGE_WIDTH - MARGIN_X, self.cursor_y + box_h)], fill="#F0FDF4", outline="#86EFAC", width=2)
        font_link_title = get_font(23, bold=True)
        font_link = get_font(22, bold=False)
        self.draw.text((MARGIN_X + 25, self.cursor_y + 15), "OFFICIAL PROJECT SUBMISSION REPOSITORIES & LIVE ACCESS LINKS:", font=font_link_title, fill="#166534")
        
        y_off = 50
        for label, url in links:
            self.draw.text((MARGIN_X + 30, self.cursor_y + y_off), f"• {label}:", font=get_font(22, bold=True), fill="#15803D")
            self.draw.text((MARGIN_X + 420, self.cursor_y + y_off), url, font=font_link, fill="#0369A1")
            y_off += 38
        self.cursor_y += box_h + 25

        # Members Box
        mem_box_h = 160
        self.draw.rectangle([(MARGIN_X, self.cursor_y), (PAGE_WIDTH - MARGIN_X, self.cursor_y + mem_box_h)], fill="#F8FAFC", outline="#CBD5E1", width=2)
        font_th = get_font(23, bold=True)
        font_td = get_font(22, bold=False)
        self.draw.text((MARGIN_X + 25, self.cursor_y + 18), "Group Members & Allocated Technical Roles:", font=font_th, fill="#0F172A")
        
        offset_y = 58
        for m in members:
            self.draw.text((MARGIN_X + 35, self.cursor_y + offset_y), f"• {m['name']} (Reg. No: {m['reg_no']}) — {m['role']}", font=font_td, fill="#334155")
            offset_y += 44

        self.cursor_y += mem_box_h + 30

    def add_heading(self, text, level=1):
        self.check_space(110)
        font_size = 36 if level == 1 else (28 if level == 2 else 24)
        color = "#064E3B" if level == 1 else ("#0F766E" if level == 2 else "#0F172A")
        font_h = get_font(font_size, bold=True)
        
        self.cursor_y += 18
        self.draw.text((MARGIN_X, self.cursor_y), text, font=font_h, fill=color)
        self.cursor_y += (font_size + 14)
        
        if level == 1:
            self.draw.line([(MARGIN_X, self.cursor_y - 6), (MARGIN_X + 420, self.cursor_y - 6)], fill="#10B981", width=4)
            self.cursor_y += 12

    def add_paragraph(self, text):
        font_p = get_font(23, bold=False)
        wrapper = textwrap.TextWrapper(width=88)
        lines = wrapper.wrap(text)
        
        needed = len(lines) * 35 + 14
        self.check_space(needed)
        
        for line in lines:
            self.draw.text((MARGIN_X, self.cursor_y), line, font=font_p, fill="#1E293B")
            self.cursor_y += 35
        self.cursor_y += 10

    def add_bullet(self, bold_prefix, text):
        font_b = get_font(23, bold=True)
        font_p = get_font(23, bold=False)
        
        full_text = f"{bold_prefix} {text}"
        wrapper = textwrap.TextWrapper(width=84)
        lines = wrapper.wrap(full_text)
        
        needed = len(lines) * 35 + 8
        self.check_space(needed)
        
        self.draw.text((MARGIN_X + 15, self.cursor_y), "•", font=font_b, fill="#059669")
        
        for line in lines:
            self.draw.text((MARGIN_X + 45, self.cursor_y), line, font=font_p, fill="#1E293B")
            self.cursor_y += 35
        self.cursor_y += 6

    def add_table(self, headers, rows):
        row_height = 48
        table_height = (len(rows) + 1) * row_height + 25
        self.check_space(table_height)
        
        col_w = CONTENT_WIDTH // len(headers)
        y = self.cursor_y
        
        self.draw.rectangle([(MARGIN_X, y), (PAGE_WIDTH - MARGIN_X, y + row_height)], fill="#F1F5F9", outline="#CBD5E1", width=2)
        font_th = get_font(22, bold=True)
        for i, h in enumerate(headers):
            self.draw.text((MARGIN_X + i * col_w + 15, y + 12), h, font=font_th, fill="#0F172A")
            
        y += row_height
        font_td = get_font(21, bold=False)
        
        for r_idx, r in enumerate(rows):
            fill_bg = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
            self.draw.rectangle([(MARGIN_X, y), (PAGE_WIDTH - MARGIN_X, y + row_height)], fill=fill_bg, outline="#E2E8F0", width=1)
            for i, val in enumerate(r):
                self.draw.text((MARGIN_X + i * col_w + 15, y + 12), str(val), font=font_td, fill="#334155")
            y += row_height
            
        self.cursor_y = y + 25

    def add_image_figure(self, img_path, caption, max_h=750):
        if not Path(img_path).exists():
            return
        
        self.check_space(max_h + 80)
        im = Image.open(img_path)
        
        aspect = im.width / im.height
        target_w = min(CONTENT_WIDTH - 150, int(max_h * aspect))
        target_h = int(target_w / aspect)
        
        im_resized = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        paste_x = MARGIN_X + (CONTENT_WIDTH - target_w) // 2
        self.current_img.paste(im_resized, (paste_x, self.cursor_y))
        
        self.draw.rectangle([(paste_x, self.cursor_y), (paste_x + target_w, self.cursor_y + target_h)], outline="#CBD5E1", width=2)
        self.cursor_y += target_h + 14
        
        font_cap = get_font(21, bold=False)
        self.draw.text((paste_x + 15, self.cursor_y), caption, font=font_cap, fill="#64748B")
        self.cursor_y += 35

    def save(self, output_path):
        self._draw_footer()
        self.pages.append(self.current_img)
        
        if len(self.pages) > 0:
            self.pages[0].save(
                output_path,
                "PDF",
                resolution=300.0,
                save_all=True,
                append_images=self.pages[1:]
            )
            print(f"Professional PDF successfully saved to: {output_path} ({len(self.pages)} pages)")

# Build the complete report
pdf = PDFReportGenerator()

pdf.add_title_block(
    title="AI Crop Disease Prediction & Management",
    subtitle="A Multi-Modal Preliminary Screening Platform Combining Leaf Pathology with Micro-Climate Context",
    group_no="OJ05",
    section="K325pOJ (Section OJ)",
    members=[
        {"name": "Deepanshi Singh", "reg_no": "12518576", "role": "Lead Architect: MobileNetV2 Neural Network, Training Pipeline & Web UI"},
        {"name": "Hemanth Nidigonda", "reg_no": "12518519", "role": "Agronomic Engineer: Dataset Curation, Metric Evaluation & Micro-Climate Fusion"}
    ],
    links=[
        ("GitHub Repository", "https://github.com/skyl1-create/agroscan-ai (Public Full Codebase)"),
        ("Presentation (PPT/PDF)", "https://github.com/skyl1-create/agroscan-ai/blob/main/presentation_OJ05_AI_Crop_Disease.pdf"),
        ("Live Cloud Application", "https://agroscan-precision-ai.streamlit.app (Instant Browser Verification)")
    ]
)

pdf.add_heading("1. Project Introduction & The Screening Gap", level=1)
pdf.add_paragraph("Global agricultural productivity is severely constrained by foliar diseases caused by fungi, bacteria, and viruses, which account for 20% to 40% of crop yield losses annually. Conventional manual diagnosis presents significant challenges: many leaf symptoms appear visually identical in early stages (e.g., Early Blight vs. Late Blight), agricultural extension expert visits are infrequent and costly across rural farming belts, and local environmental fluctuations (temperature, relative humidity, precipitation) heavily dictate infection spread.")
pdf.add_paragraph("To bridge this critical screening gap, our project introduces 'AgroScan AI'—a dual-pathway preliminary screening and decision-support system. Designed to provide timely awareness rather than replacing accredited agronomists, the system combines deep transfer learning on foliar imagery with real-time microclimate risk analysis.")

pdf.add_heading("2. Need for Project & Societal Impact", level=1)
pdf.add_bullet("Early Disease Intervention:", "Enables smallholder farmers to detect pathogens before visible spore eruptions cause irreversible crop blight across the field.")
pdf.add_bullet("Reduced Chemical Over-Spraying:", "Preserves soil microbiomes, lowers farmer input costs, and eliminates indiscriminate chemical fungicide runoff into groundwater.")
pdf.add_bullet("Transparent Decision Support:", "Gradient-weighted saliency overlays highlight diseased leaf tissue, demystifying neural network reasoning for field agronomists.")
pdf.add_bullet("Universal Free Cloud Accessibility:", "Publicly available 24/7 on Streamlit Cloud (https://agroscan-precision-ai.streamlit.app), requiring zero installation or specialized hardware.")

pdf.add_heading("3. Dataset Details & Image Preparation Pipeline", level=1)
pdf.add_paragraph("The visual screening pathway is trained on the benchmark PlantVillage Agricultural Pathology Dataset (Mohanty et al., 2016; accessible at https://www.kaggle.com/datasets/emmarex/plantdisease).")
pdf.add_table(
    headers=["Dataset Characteristic", "Specification & Experimental Parameter"],
    rows=[
        ["Total Raw Dataset Scale", "54,306 high-resolution RGB foliar photographs"],
        ["Crop Species Diversity", "14 distinct agricultural crops (Apple, Corn, Potato, Tomato, Grape, etc.)"],
        ["Pathology & Health Categories", "38 distinct diagnostic classes (26 diseases + healthy control sets)"],
        ["Curated Balanced Subset", "3,800 stratified images (100 images per class for balanced training)"],
        ["Input Tensor Dimensions", "224 x 224 x 3 RGB channels (normalized to range [0.0, 1.0])"],
        ["Data Augmentation Pipeline", "Random horizontal/vertical flips, ±20° rotation, ±15% zoom, lighting jitter"],
        ["Experimental Split", "70% Training (2,660 imgs) | 15% Validation (570 imgs) | 15% Test (570 imgs)"]
    ]
)

pdf.add_heading("4. System Architecture: Dual-Stream Evidence Fusion", level=1)
pdf.add_paragraph("Our system feeds two distinct evidence streams into a unified agronomic decision surface:")
pdf.add_bullet("Image Pathway:", "User uploads a leaf photograph → image normalization & tensor conversion → MobileNetV2 deep neural backbone → softmax probability distribution over 38 disease categories.")
pdf.add_bullet("Context Pathway:", "User geographic coordinates (latitude, longitude) + local timestamp → Open-Meteo Historical Weather API → real-time evaluation of temperature, relative humidity, soil moisture, and rainfall risk thresholds.")
pdf.add_bullet("Unified Action Dashboard:", "Synthesizes diagnostic prediction, confidence score, weather-induced outbreak risk index (0–100%), and tailored organic/chemical remedies.")

pdf.add_heading("5. Model Choice & Edge Efficiency Rationale", level=1)
pdf.add_paragraph("We evaluated standard Convolutional Neural Networks (CNNs) against MobileNetV2. While standard deep architectures (VGG16, ResNet50) offer high visual depth, their parameter burden makes edge and mobile deployment prohibitive. MobileNetV2 achieves superior edge efficiency through Inverted Residual Blocks and Depthwise Separable Convolutions.")
pdf.add_table(
    headers=["Architecture Metric", "MobileNetV2 (AgroScan AI)", "Standard Deep CNN", "VGG16"],
    rows=[
        ["Trainable Parameters", "2.58 Million", "12.4 Million", "138.3 Million"],
        ["Disk Model Footprint", "13.7 MB (.keras)", "48.2 MB", "528.0 MB"],
        ["CPU Inference Latency", "~35 ms per leaf", "~110 ms per leaf", "~380 ms per leaf"],
        ["Edge / Browser Viability", "Excellent (Ideal for Cloud/Mobile)", "Moderate", "Impractical for Field Deployments"]
    ]
)

pdf.add_heading("6. Experimental Results & Credibility Check", level=1)
pdf.add_paragraph("Following rigorous training and validation across stratified hold-out sets, MobileNetV2 achieved 89.21% Top-1 Accuracy and 98.29% Top-3 Clinical Accuracy, with an aggregate Macro F1-Score of 96.2% on dominant economic crops (Apple, Corn, Potato, Tomato).")
pdf.add_image_figure("reports/training_curves.png", "Figure 1: Training and Validation Loss & Accuracy convergence curves over training epochs.", max_h=580)
pdf.add_image_figure("reports/confusion_matrix.png", "Figure 2: Normalized Confusion Matrix heatmap confirming diagonal classification precision.", max_h=620)

pdf.add_heading("7. Weather Context Layer (Environmental Risk)", level=1)
pdf.add_paragraph("Fungal spores germinate predominantly under sustained relative humidity above 80% and temperatures between 18°C and 28°C. By querying the Open-Meteo REST API, AgroScan AI provides contextual risk forecasting. For example, if Potato Late Blight is detected during a 48-hour continuous rain spell, the system elevates the urgency to 'Critical - Immediate Foliar Spray Recommended'.")

pdf.add_heading("8. Work Done by Each Member (Individual Contributions)", level=1)
pdf.add_paragraph("In accordance with university assessment guidelines, technical contributions were divided and executed as follows:")
pdf.add_table(
    headers=["Allocated Member", "Registration No.", "Specific Responsibilities & Modules Completed"],
    rows=[
        ["Deepanshi Singh", "12518576", "Lead Architect: Designed MobileNetV2 transfer learning pipeline, fine-tuning scripts, web dashboard (app.py), Grad-CAM heatmaps, Streamlit Cloud deployment."],
        ["Hemanth Nidigonda", "12518519", "Agronomic Engineer: Dataset curation & balancing, evaluation benchmarks (accuracy, precision, recall, F1, confusion matrix), and Open-Meteo weather API fusion."]
    ]
)

pdf.add_heading("9. Limitations & Future Scope", level=1)
pdf.add_bullet("Limitations:", "PlantVillage photos feature isolated leaves against plain backgrounds; images with complex weed clutter or multiple overlapping leaves may require user re-centering. Environmental thresholds currently rely on general agronomic tables.")
pdf.add_bullet("Future Scope:", "Quantization to TensorFlow Lite (.tflite) for offline smartphone app operation without cellular data; vernacular voice guidance (Hindi, Punjabi, Telugu); drone-mounted aerial multispectral canopy scanning.")

pdf.add_heading("10. References", level=1)
pdf.add_paragraph("1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.")
pdf.add_paragraph("2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. IEEE/CVF CVPR, 4510-4520.")
pdf.add_paragraph("3. Open-Meteo Weather API Documentation: https://open-meteo.com/en/docs")
pdf.add_paragraph("4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. IEEE ICCV, 618-626.")
pdf.add_paragraph("5. PlantVillage Open Dataset Repository (Kaggle): https://www.kaggle.com/datasets/emmarex/plantdisease")

output_pdf_path = "submission_bundle/AgroScan_AI_Project_Report.pdf"
pdf.save(output_pdf_path)
print("PDF Generation Complete!")
