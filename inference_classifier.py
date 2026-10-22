"""
Sign Language Detector - Unified ASL Manual Alphabet (A-Z) & WLASL
Runs real-time webcam detection with Wise Camera framing guidance.
Press 'Q' to quit | 'M' to switch mode (Alphabet / WLASL / All) | 'C' to clear
"""

from inference_engine import run_desktop_live

if __name__ == "__main__":
    run_desktop_live()
