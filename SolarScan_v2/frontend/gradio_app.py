"""
gradio_app.py — Frontend สำหรับ SolarScan Classification
"""
import gradio as gr
import requests
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

def classify_solar(image):
    if image is None:
        return "กรุณาอัปโหลดภาพ", 0.0, 0.0

    # แปลงภาพ → bytes
    pil_image = Image.fromarray(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)

    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=30
        )
        result = response.json()

        has_solar  = result["has_solar"]
        confidence = result["confidence"] * 100
        solar_prob = result["probabilities"]["solar"] * 100

        if has_solar:
            label = f"✅ มีแผงโซลาร์ (มั่นใจ {confidence:.1f}%)"
        else:
            label = f"❌ ไม่มีแผงโซลาร์ (มั่นใจ {confidence:.1f}%)"

        return label, round(solar_prob, 1), round(100 - solar_prob, 1)

    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, 0.0

# ── Gradio UI ──
with gr.Blocks(title="SolarScan") as demo:

    gr.Markdown("""
    # 🛰️ SolarScan
    ### ตรวจจับแผงโซลาร์จากภาพดาวเทียมด้วย AI
    อัปโหลดภาพดาวเทียม แล้วให้ AI บอกว่ามีแผงโซลาร์หรือไม่
    """)

    with gr.Row():
        # ── Input ──
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📁 อัปโหลดภาพดาวเทียม",
                type="numpy",
                height=300
            )
            classify_btn = gr.Button(
                "🔍 วิเคราะห์ภาพ",
                variant="primary",
                size="lg"
            )

        # ── Output ──
        with gr.Column(scale=1):
            result_text = gr.Textbox(
                label="🎯 ผลการวิเคราะห์",
                lines=2,
                interactive=False
            )
            with gr.Row():
                solar_prob    = gr.Number(label="☀️ โอกาสมีแผง (%)")
                no_solar_prob = gr.Number(label="🚫 โอกาสไม่มีแผง (%)")

    classify_btn.click(
        fn=classify_solar,
        inputs=image_input,
        outputs=[result_text, solar_prob, no_solar_prob]
    )

    gr.Markdown("""
    ---
    **SDG 7: Affordable and Clean Energy** | SolarScan v1.0
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
