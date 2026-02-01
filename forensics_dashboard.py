"""
Multi-Modal Misinformation Forensic Dashboard
Interactive Gradio web interface for analyzing content authenticity.
"""

import gradio as gr
import os
import json
from PIL import Image
from misinfo_forensics import MisinfoForensics
import subprocess
import numpy as np


_WHISPER_MODEL = None


def _extract_transcript(video_path: str) -> str:
    """Extract a speech transcript from a video using Whisper (optional dependency).

    Returns transcript text on success.
    Returns an empty string if Whisper isn't available.
    Returns a user-facing error message string if transcription fails.
    """
    if not video_path:
        return ""

    if not os.path.exists(video_path):
        return f"[Video file not found: {video_path}]"

    try:
        import whisper  # type: ignore
    except Exception:
        return "[Whisper not installed: pip install openai-whisper (and install ffmpeg)]"

    # Use bundled ffmpeg from imageio-ffmpeg to avoid relying on a system ffmpeg.
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None

    if not ffmpeg_exe or not os.path.exists(str(ffmpeg_exe)):
        return "[Missing bundled ffmpeg: pip install imageio-ffmpeg]"

    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        _WHISPER_MODEL = whisper.load_model(model_name)

    try:
        # Decode audio to 16kHz mono PCM using bundled ffmpeg.
        # This avoids Whisper's internal `ffmpeg` subprocess lookup (common WinError 2 cause).
        cmd = [
            str(ffmpeg_exe),
            "-nostdin",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not proc.stdout:
            err = (proc.stderr.decode("utf-8", errors="ignore") or "").strip()
            err = err[-800:] if err else ""
            return f"[Audio decode failed via ffmpeg: {err or 'unknown error'}]"

        audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return "[No audio stream detected in video]"

        # fp16=False for CPU compatibility; Whisper accepts audio arrays.
        try:
            result = _WHISPER_MODEL.transcribe(audio, fp16=False)
        except Exception:
            # Older APIs expose a module-level transcribe() helper.
            result = whisper.transcribe(_WHISPER_MODEL, audio, fp16=False)

        return (result.get("text") or "").strip()
    except Exception as e:
        return f"[Transcript extraction failed: {e}]"

# Initialize the forensics system globally
print("🚀 Initializing Forensic System...")
forensics = MisinfoForensics(
    fusion_weights="forensics_master_final.pth",
    faiss_index_path="guardian_embeddings.pkl"
)
print("✓ System ready!")


def _probs(label_a: str, prob_a: float, label_b: str, prob_b: float) -> dict:
    """Return a Gradio `Label`-compatible probability mapping."""
    try:
        prob_a = float(prob_a)
    except Exception:
        prob_a = 0.0
    try:
        prob_b = float(prob_b)
    except Exception:
        prob_b = 0.0

    prob_a = max(0.0, min(1.0, prob_a))
    prob_b = max(0.0, min(1.0, prob_b))
    total = prob_a + prob_b
    if total <= 0:
        return {label_a: 0.5, label_b: 0.5}
    return {label_a: prob_a / total, label_b: prob_b / total}


def _verdict_badge_html(verdict_text: str, confidence: float) -> str:
    verdict_text = (verdict_text or "").upper().strip()
    confidence = float(confidence) if confidence is not None else 0.0
    if verdict_text == "FAKE":
        return f"<div class='verdict-fake'>🔴 FAKE<br><span style='font-size:0.75em;font-weight:600'>{confidence:.1%} confidence</span></div>"
    if verdict_text == "REAL":
        return f"<div class='verdict-real'>🟢 REAL<br><span style='font-size:0.75em;font-weight:600'>{confidence:.1%} confidence</span></div>"
    return "<div class='verdict-neutral'>⚪ UNKNOWN</div>"


def _normalize_video_path(video_value):
    """Gradio versions return video as str path or dict; normalize to path."""
    if video_value is None:
        return None
    if isinstance(video_value, str):
        p = video_value.strip()
        return p or None
    if isinstance(video_value, dict):
        # Common Gradio formats
        p = video_value.get('path') or video_value.get('name') or video_value.get('video')
        if isinstance(p, str):
            p = p.strip()
            return p or None
    return None


def predict(image_path, user_caption, video_value):
    """
    Run complete forensic analysis on user-submitted content.
    
    Args:
        image_path: Path to uploaded image
        user_caption: User's text/headline
    
    Returns:
        Tuple of all UI component values
    """
    caption = (user_caption or "").strip()
    video_path = _normalize_video_path(video_value)
    transcript_text = ""
    if video_path:
        transcript_text = _extract_transcript(video_path)

    combined_text = caption
    if transcript_text and not transcript_text.startswith("["):
        combined_text = (caption + "\n\n" + transcript_text).strip() if caption else transcript_text
    has_text = bool(caption)
    has_visual = bool(image_path) or bool(video_path)

    if image_path is None and not caption and video_path is None:
        na = {"N/A": 1.0}
        return (
            "",  # Verdict badge HTML
            {"WAITING": 1.0},
            na, na, na, na, na,  # Scores
            None,  # Gallery
            "### Missing Input\nPlease provide text, an image, or a video.",
            "",  # Vault metadata
            ""  # Transcript
        )
    
    try:
        # Run forensic analysis
        results = forensics.analyze(
            text=combined_text if combined_text else None,
            image_path=image_path,
            video_path=video_path,
            verbose=False
        )
        
        scores = results['scores']
        verdict = results['verdict_text']
        confidence = results['confidence']
        vault_matches = results.get('vault_matches', [])
        explanation = results['explanation']
        
        # 1. Verdict Badge + Label probabilities
        fake_prob = float(scores.get('fake_probability', 0.0))
        real_prob = float(scores.get('real_probability', 1.0 - fake_prob))
        verdict_badge = _verdict_badge_html(verdict, confidence)
        verdict_output = _probs("REAL", real_prob, "FAKE", fake_prob)
        
        # 2. Individual Forensic Scores
        na = {"N/A": 1.0}

        if has_text:
            ai_p = float(scores.get('ai_score', 0.0))
            ai_score = _probs("AI-generated", ai_p, "Human-written", 1.0 - ai_p)
            misinfo_p = float(scores.get('misinfo_score', 0.0))
            misinfo_score = _probs("Suspicious", misinfo_p, "Normal", 1.0 - misinfo_p)
        else:
            ai_score = na
            misinfo_score = na

        if has_visual:
            deepfake_p = float(scores.get('deepfake_score', 0.0))
            deepfake_score = _probs("Manipulated", deepfake_p, "Authentic", 1.0 - deepfake_p)
            vault_p = float(scores.get('vault_discrepancy', 0.0))
            vault_discrepancy = _probs("Archive match", vault_p, "No match", 1.0 - vault_p)
        else:
            deepfake_score = na
            vault_discrepancy = na

        if has_text and has_visual:
            clip_sim = float(scores.get('clip_similarity', 0.0))
            clip_norm = (clip_sim + 1.0) / 2.0  # [-1,1] -> [0,1]
            clip_norm = max(0.0, min(1.0, clip_norm))
            clip_score = _probs("Aligned", clip_norm, "Misaligned", 1.0 - clip_norm)
        else:
            clip_score = na
        
        # 3. Truth Vault Gallery (top matches)
        gallery_images = []
        vault_metadata = ""
        
        if not has_visual:
            vault_metadata = """### 🗃️ Truth Vault Cross-Check

**Skipped**

Upload an image or a video to enable archive matching.
"""
        elif vault_matches and len(vault_matches) > 0:
            # Get top match details
            top_match = vault_matches[0]
            
            # Try to load the matched image if it exists
            if top_match.get('url') and os.path.exists(top_match['url']):
                gallery_images = [(top_match['url'], f"{top_match['similarity']:.1%} Match")]
            
            # Build metadata markdown
            vault_metadata = f"""### 🗃️ Truth Vault Cross-Check

**Top Match Found:**
- **Original Headline:** "{top_match['title']}"
- **Image Similarity:** {top_match['similarity']:.1%}
- **Text Similarity:** {scores.get('text_similarity', 0):.1%}
- **Published:** {top_match.get('date', 'N/A')}
- **Semantic Mismatch:** {abs(scores.get('text_similarity', 0) - top_match['similarity']):.1%}

{"⚠️ **Warning:** This image was previously used in a different context!" if top_match['similarity'] > 0.85 else "✓ No significant archive matches found."}
"""
        else:
            vault_metadata = """### 🗃️ Truth Vault Cross-Check

**No Archive Matches Found**

No image/video match found in our Guardian database of 2,170 verified articles.
"""
        
        # 4. Forensic Summary (Gemini explanation)
        forensic_summary = f"""### 📊 Forensic Analysis Summary

{explanation}

---

**Detailed Metrics:**
- **Final Verdict:** {verdict} ({confidence:.1%} confidence)
- **REAL Probability:** {scores.get('real_probability', 0):.2%}
- **FAKE Probability:** {scores.get('fake_probability', 0):.2%}

**Individual Signals:**
- AI-Generated Text: {scores.get('ai_score', 0.0):.2%}
- Propaganda/Misinfo: {scores.get('misinfo_score', 0.0):.2%}
- Deepfake Visual: {scores.get('deepfake_score', 0.0):.2%}
- CLIP Consistency: {scores.get('clip_similarity', 0.0):.4f}
- Archive Discrepancy: {scores.get('vault_discrepancy', 0.0):.2%}
"""
        
        return (
            verdict_badge,
            verdict_output,
            ai_score,
            misinfo_score,
            deepfake_score,
            clip_score,
            vault_discrepancy,
            gallery_images if gallery_images else None,
            forensic_summary,
            vault_metadata,
            transcript_text
        )
        
    except Exception as e:
        error_msg = f"""### ❌ Analysis Error

**Error:** {str(e)}

Please ensure:
1. Image file is valid (JPG/PNG)
2. Caption is provided
3. All model weights are loaded correctly
"""
        na = {"N/A": 1.0}
        return (
            "<div class='verdict-neutral'>❌ ERROR</div>",
            {"ERROR": 1.0},
            na, na, na, na, na,  # Scores
            None,  # Gallery
            error_msg,
            "",  # Vault metadata
            transcript_text or ""  # Transcript
        )


# Custom CSS for branding
custom_css = """
/* Main container */
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px !important;
}

/* Header styling */
h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    margin-bottom: 0.5em;
}

/* Verdict Badge Styling */
.verdict-real {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.5em !important;
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3) !important;
}

.verdict-fake {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.5em !important;
    padding: 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3) !important;
}

.verdict-neutral {
    background: #3a3a3a !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.5em !important;
    padding: 20px !important;
    border-radius: 12px !important;
}

/* Score cards */
.score-card {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Gallery styling */
.gallery {
    border: 2px solid #667eea;
    border-radius: 12px;
    padding: 10px;
}

/* Button styling */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    padding: 12px 30px !important;
    border-radius: 8px !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Input boxes */
.gr-input, .gr-textbox {
    border: 2px solid #e0e0e0 !important;
    border-radius: 8px !important;
}

.gr-input:focus, .gr-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}
"""

# Build the Gradio interface
with gr.Blocks(css=custom_css, title="Misinformation Forensics Lab") as demo:
    
    # Header
    gr.Markdown("""
    # 🔍 Multi-Modal Misinfo Forensic Lab
    ### AI-Powered Content Authenticity Analysis
    Provide text, an image, or a video (optionally with text) to run comprehensive forensic analysis using RoBERTa, EfficientNet, CLIP, and our Guardian Truth Vault.
    """)
    
    with gr.Row():
        # Left: Input Components
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Submit Content for Analysis")
            image_input = gr.Image(
                type='filepath',
                label="Upload Image",
                height=300
            )
            video_input = gr.Video(
                label="Upload Video"
            )
            text_input = gr.Textbox(
                label="User Caption / Headline",
                placeholder="Enter the text accompanying this image...",
                lines=3
            )
            transcript_output = gr.Textbox(
                label="🎙️ Video Transcript (Whisper)",
                placeholder="Transcript will appear here when you upload a video.",
                lines=6,
                interactive=False
            )
            analyze_btn = gr.Button("🔬 Run Forensic Analysis", variant="primary", size="lg")
        
        # Right: Verdict Badge
        with gr.Column(scale=1):
            gr.Markdown("### ⚖️ Final Verdict")
            verdict_badge = gr.HTML("", label="Verdict Badge")
            verdict_output = gr.Label(
                label="Forensic Verdict",
                num_top_classes=1,
                show_label=False
            )
    
    gr.Markdown("---")
    
    # Mid Section: Forensic Scoreboard
    gr.Markdown("## 📊 Forensic Scoreboard")
    gr.Markdown("*Individual detective scores from each analysis module*")
    
    with gr.Row():
        ai_text_score = gr.Label(label="🤖 AI Text Detection", num_top_classes=1)
        misinfo_score_output = gr.Label(label="📰 Propaganda Analysis", num_top_classes=1)
        deepfake_score_output = gr.Label(label="🎭 Deepfake Detection", num_top_classes=1)
    
    with gr.Row():
        clip_score_output = gr.Label(label="🔗 Image-Text Consistency", num_top_classes=1)
        vault_score_output = gr.Label(label="🗃️ Archive Match", num_top_classes=1)
    
    gr.Markdown("---")
    
    # Bottom Row: Truth Vault
    gr.Markdown("## 🗃️ Truth Vault Analysis")
    
    with gr.Row():
        with gr.Column(scale=1):
            vault_gallery = gr.Gallery(
                label="Archive Matches",
                show_label=True,
                columns=1,
                rows=1,
                height=300,
                object_fit="contain"
            )
        
        with gr.Column(scale=1):
            vault_metadata_output = gr.Markdown("")
    
    gr.Markdown("---")
    
    # Final Report
    gr.Markdown("## 📝 Forensic Report")
    forensic_summary_output = gr.Markdown("")
    
    # Footer
    gr.Markdown("""
    ---
    **Powered by:** RoBERTa (dual heads) • EfficientNet-B0 • CLIP ViT-B/32 • Fusion Layer • Guardian Truth Vault (2,170 articles) • Gemini AI
    
    **Model Performance:** Fusion Judge trained to 95.38% accuracy (Epoch 8)
    """)
    
    # Connect the analyze button
    analyze_btn.click(
        fn=predict,
        inputs=[image_input, text_input, video_input],
        outputs=[
            verdict_badge,
            verdict_output,
            ai_text_score,
            misinfo_score_output,
            deepfake_score_output,
            clip_score_output,
            vault_score_output,
            vault_gallery,
            forensic_summary_output,
            vault_metadata_output,
            transcript_output
        ]
    )
    
    # Add examples
    gr.Markdown("## 📚 Example Cases")
    gr.Examples(
        examples=[
            [
                "guardian_processed/https___motivatedgrammar.wordpress.com_2007_11_23_til-v-till-v-til-v-until_-200.jpg",
                "Breaking news: Major political scandal uncovered today",
                None
            ],
        ],
        inputs=[image_input, text_input, video_input],
        outputs=[
            verdict_badge,
            verdict_output,
            ai_text_score,
            misinfo_score_output,
            deepfake_score_output,
            clip_score_output,
            vault_score_output,
            vault_gallery,
            forensic_summary_output,
            vault_metadata_output,
            transcript_output
        ],
        fn=predict,
        cache_examples=False
    )


if __name__ == "__main__":
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    try:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=preferred_port,
            show_error=True
        )
    except OSError:
        # If the preferred port is taken, fall back to an available ephemeral port.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            fallback_port = int(s.getsockname()[1])
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=fallback_port,
            show_error=True
        )



