"""
Train Unified ASL Alphabet (A-Z) and Word-Level ASL (WLASL) Model
Total Lexicon: 42 classes (26 ASL Alphabet Letters + 16 Core WLASL Words)
With scale, rotation, translation, bilateral mirroring, and noise invariance.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json

MODEL_FILE = "phrase_model_v2.p"
BENCHMARK_FILE = "benchmark_results.json"

ALPHABET_LEXICON = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

WLASL_LEXICON = [
    "Hello", "Thank you", "Please", "Sorry", "Yes", "No",
    "Help", "Good", "Bad", "I love you", "Want", "More",
    "Eat", "Stop", "Friend", "Call me"
]

ALL_CLASSES = sorted(list(set(ALPHABET_LEXICON + WLASL_LEXICON)))

def create_base_landmarks(sign_name):
    """
    Returns (21, 3) numpy array for authentic ASL phonological handshape.
    Landmarks:
      0: Wrist (0.5, 0.75, 0.0)
      Thumb: 1(cmc), 2(mcp), 3(ip), 4(tip)
      Index: 5(mcp), 6(pip), 7(dip), 8(tip)
      Middle: 9(mcp), 10(pip), 11(dip), 12(tip)
      Ring: 13(mcp), 14(pip), 15(dip), 16(tip)
      Pinky: 17(mcp), 18(pip), 19(dip), 20(tip)
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist origin
    pts[0] = [0.5, 0.75, 0.0]

    # Palm MCP arch
    pts[1]  = [0.46, 0.69, 0.0]   # Thumb CMC
    pts[2]  = [0.43, 0.63, 0.0]   # Thumb MCP
    pts[5]  = [0.45, 0.55, 0.0]   # Index MCP
    pts[9]  = [0.50, 0.54, 0.0]   # Middle MCP
    pts[13] = [0.54, 0.55, 0.0]   # Ring MCP
    pts[17] = [0.58, 0.58, 0.0]   # Pinky MCP

    # Helper functions for finger postures
    def set_straight(mcp_idx, tip_idx, x_pos, y_start=0.55, length=0.28):
        # Finger pointing straight up
        pts[mcp_idx + 1] = [x_pos, y_start - length * 0.35, 0.0]
        pts[mcp_idx + 2] = [x_pos, y_start - length * 0.70, 0.0]
        pts[tip_idx]     = [x_pos, y_start - length * 1.00, 0.0]

    def set_curled_fist(mcp_idx, tip_idx, x_pos, z_depth=-0.04):
        # Finger fully curled into palm fist
        pts[mcp_idx + 1] = [x_pos, 0.60, z_depth]
        pts[mcp_idx + 2] = [x_pos, 0.65, z_depth * 0.5]
        pts[tip_idx]     = [x_pos, 0.63, 0.02]

    def set_curved_c(mcp_idx, tip_idx, x_pos):
        # Curved in C-shape
        pts[mcp_idx + 1] = [x_pos, 0.48, 0.06]
        pts[mcp_idx + 2] = [x_pos, 0.51, 0.12]
        pts[tip_idx]     = [x_pos, 0.56, 0.13]

    def set_hooked(mcp_idx, tip_idx, x_pos):
        # Hooked (PIP bent 90 deg, DIP curled down)
        pts[mcp_idx + 1] = [x_pos, 0.47, 0.03]
        pts[mcp_idx + 2] = [x_pos, 0.53, 0.07]
        pts[tip_idx]     = [x_pos, 0.57, 0.05]

    # --- ASL ALPHABET (A-Z) ---
    if sign_name == "A":
        # Fist with thumb pointing straight UP resting alongside index MCP/PIP
        pts[3] = [0.41, 0.52, 0.02]
        pts[4] = [0.41, 0.44, 0.03] # Thumb pointing up alongside index
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "B":
        # 4 fingers straight UP, thumb folded across palm
        pts[3] = [0.47, 0.64, 0.03]
        pts[4] = [0.51, 0.65, 0.04] # Thumb folded across base of fingers
        set_straight(5, 8, 0.45)
        set_straight(9, 12, 0.50)
        set_straight(13, 16, 0.54)
        set_straight(17, 20, 0.58)

    elif sign_name == "C":
        # All 5 fingers curved forming 'C' shape
        pts[3] = [0.38, 0.60, 0.08]
        pts[4] = [0.40, 0.56, 0.12]
        set_curved_c(5, 8, 0.45)
        set_curved_c(9, 12, 0.50)
        set_curved_c(13, 16, 0.54)
        set_curved_c(17, 20, 0.57)

    elif sign_name == "D":
        # Index straight UP; thumb tip touching middle, ring, pinky tips forming circle
        pts[3] = [0.47, 0.57, 0.04]
        pts[4] = [0.50, 0.55, 0.05] # Thumb touching middle/ring tips
        set_straight(5, 8, 0.45)    # Index straight up
        set_hooked(9, 12, 0.50)
        set_hooked(13, 16, 0.53)
        set_hooked(17, 20, 0.56)

    elif sign_name == "E":
        # All 4 fingers curled down, thumb tucked horizontally below fingertips
        pts[3] = [0.46, 0.60, 0.03]
        pts[4] = [0.52, 0.60, 0.04] # Thumb tucked under tips
        pts[6] = [0.45, 0.50, 0.02]; pts[7] = [0.46, 0.56, 0.03]; pts[8] = [0.46, 0.58, 0.01]
        pts[10] = [0.50, 0.49, 0.02]; pts[11] = [0.50, 0.55, 0.03]; pts[12] = [0.50, 0.57, 0.01]
        pts[14] = [0.54, 0.50, 0.02]; pts[15] = [0.54, 0.56, 0.03]; pts[16] = [0.54, 0.58, 0.01]
        pts[18] = [0.58, 0.53, 0.02]; pts[19] = [0.58, 0.58, 0.03]; pts[20] = [0.58, 0.60, 0.01]

    elif sign_name == "F":
        # Thumb & index tips touching (forming circle); middle, ring, pinky straight up
        pts[3] = [0.44, 0.55, 0.03]
        pts[4] = [0.46, 0.50, 0.04] # Thumb touching index tip
        pts[6] = [0.45, 0.48, 0.02]; pts[7] = [0.46, 0.49, 0.03]; pts[8] = [0.46, 0.50, 0.04]
        set_straight(9, 12, 0.51)
        set_straight(13, 16, 0.55)
        set_straight(17, 20, 0.59)

    elif sign_name == "G":
        # Index pointing horizontally sideways, thumb parallel horizontally
        pts[3] = [0.41, 0.59, 0.02]; pts[4] = [0.36, 0.58, 0.03]
        pts[6] = [0.40, 0.55, 0.01]; pts[7] = [0.35, 0.55, 0.01]; pts[8] = [0.30, 0.55, 0.01]
        set_curled_fist(9, 12, 0.48)
        set_curled_fist(13, 16, 0.52)
        set_curled_fist(17, 20, 0.56)

    elif sign_name == "H":
        # Index AND middle fingers pointing horizontally sideways together
        pts[3] = [0.42, 0.62, 0.03]; pts[4] = [0.45, 0.62, 0.04]
        pts[6] = [0.40, 0.53, 0.01]; pts[7] = [0.35, 0.53, 0.01]; pts[8] = [0.30, 0.53, 0.01]
        pts[10] = [0.40, 0.56, 0.01]; pts[11] = [0.35, 0.56, 0.01]; pts[12] = [0.30, 0.56, 0.01]
        set_curled_fist(13, 16, 0.53)
        set_curled_fist(17, 20, 0.57)

    elif sign_name == "I":
        # Pinky pointing straight UP; other 4 fingers curled
        pts[3] = [0.47, 0.61, 0.03]; pts[4] = [0.50, 0.61, 0.04]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_straight(17, 20, 0.58)

    elif sign_name == "J":
        # Pinky up and hooked
        pts[3] = [0.47, 0.61, 0.03]; pts[4] = [0.50, 0.61, 0.04]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        pts[18] = [0.58, 0.48, 0.0]; pts[19] = [0.57, 0.40, 0.02]; pts[20] = [0.54, 0.38, 0.03]

    elif sign_name == "K":
        # Index straight up, middle finger angled forward/up at 45 deg, thumb between them
        pts[3] = [0.45, 0.54, 0.03]; pts[4] = [0.47, 0.47, 0.04] # Thumb tip pointing up between
        set_straight(5, 8, 0.45)
        pts[10] = [0.50, 0.47, 0.05]; pts[11] = [0.51, 0.40, 0.09]; pts[12] = [0.52, 0.35, 0.12]
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "L":
        # Index straight UP, thumb extended out horizontally at 90 degrees ('L' shape)
        pts[3] = [0.38, 0.62, 0.0]; pts[4] = [0.31, 0.62, 0.0] # Thumb pointing left
        set_straight(5, 8, 0.45)                               # Index straight up
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "M":
        # 3 fingers (index, middle, ring) curled over thumb
        pts[3] = [0.48, 0.63, 0.01]; pts[4] = [0.54, 0.64, 0.01] # Thumb under 3 fingers
        set_curled_fist(5, 8, 0.46, 0.04)
        set_curled_fist(9, 12, 0.50, 0.04)
        set_curled_fist(13, 16, 0.54, 0.04)
        set_curled_fist(17, 20, 0.58, -0.04)

    elif sign_name == "N":
        # 2 fingers (index, middle) curled over thumb
        pts[3] = [0.47, 0.63, 0.01]; pts[4] = [0.51, 0.64, 0.01] # Thumb under 2 fingers
        set_curled_fist(5, 8, 0.46, 0.04)
        set_curled_fist(9, 12, 0.50, 0.04)
        set_curled_fist(13, 16, 0.54, -0.04)
        set_curled_fist(17, 20, 0.58, -0.04)

    elif sign_name == "O":
        # All 5 fingers curved with tips meeting thumb tip forming 'O'
        pts[3] = [0.42, 0.57, 0.06]; pts[4] = [0.47, 0.52, 0.08] # Thumb tip
        pts[6] = [0.45, 0.46, 0.04]; pts[7] = [0.47, 0.49, 0.06]; pts[8] = [0.47, 0.52, 0.08]
        pts[10] = [0.49, 0.45, 0.04]; pts[11] = [0.49, 0.49, 0.06]; pts[12] = [0.48, 0.52, 0.08]
        pts[14] = [0.53, 0.46, 0.04]; pts[15] = [0.51, 0.49, 0.06]; pts[16] = [0.49, 0.52, 0.08]
        pts[18] = [0.57, 0.49, 0.04]; pts[19] = [0.53, 0.51, 0.06]; pts[20] = [0.50, 0.53, 0.08]

    elif sign_name == "P":
        # Downward pointing K (index horizontal, middle down, thumb between)
        pts[3] = [0.44, 0.60, 0.02]; pts[4] = [0.42, 0.64, 0.03]
        pts[6] = [0.42, 0.55, 0.0]; pts[7] = [0.38, 0.55, 0.0]; pts[8] = [0.34, 0.55, 0.0]
        pts[10] = [0.49, 0.60, 0.0]; pts[11] = [0.48, 0.66, 0.0]; pts[12] = [0.47, 0.72, 0.0]
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Q":
        # Downward pointing G (index and thumb pointing down)
        pts[3] = [0.42, 0.66, 0.0]; pts[4] = [0.41, 0.73, 0.0]
        pts[6] = [0.45, 0.62, 0.0]; pts[7] = [0.45, 0.69, 0.0]; pts[8] = [0.45, 0.76, 0.0]
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "R":
        # Index and middle fingers straight up and crossed
        pts[3] = [0.47, 0.62, 0.03]; pts[4] = [0.50, 0.62, 0.04]
        # Middle crosses in front of index
        pts[6] = [0.45, 0.46, 0.0]; pts[7] = [0.46, 0.38, 0.0]; pts[8] = [0.48, 0.30, 0.01]
        pts[10] = [0.49, 0.45, 0.0]; pts[11] = [0.47, 0.37, 0.02]; pts[12] = [0.45, 0.29, 0.02]
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "S":
        # Fist with thumb wrapped across front of fingers
        pts[3] = [0.48, 0.59, 0.05]; pts[4] = [0.53, 0.59, 0.06] # Thumb wrapped over front
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "T":
        # Thumb tucked between index and middle (index curled over thumb)
        pts[3] = [0.46, 0.55, 0.02]; pts[4] = [0.48, 0.50, 0.03] # Thumb poking between
        pts[6] = [0.45, 0.50, 0.05]; pts[7] = [0.46, 0.56, 0.04]; pts[8] = [0.46, 0.62, 0.02]
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "U":
        # Index and middle fingers straight up, pressed tightly together
        pts[3] = [0.47, 0.62, 0.03]; pts[4] = [0.50, 0.62, 0.04]
        set_straight(5, 8, 0.47, length=0.28)
        set_straight(9, 12, 0.50, length=0.28)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "V":
        # Index and middle fingers straight up, separated in 'V' shape
        pts[3] = [0.47, 0.62, 0.03]; pts[4] = [0.50, 0.62, 0.04]
        pts[6] = [0.44, 0.46, 0.0]; pts[7] = [0.42, 0.37, 0.0]; pts[8] = [0.40, 0.28, 0.0]
        pts[10] = [0.51, 0.45, 0.0]; pts[11] = [0.53, 0.36, 0.0]; pts[12] = [0.55, 0.27, 0.0]
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "W":
        # Index, middle, and ring fingers straight up in 'W' shape
        pts[3] = [0.47, 0.62, 0.03]; pts[4] = [0.51, 0.62, 0.04]
        pts[6] = [0.44, 0.46, 0.0]; pts[7] = [0.42, 0.37, 0.0]; pts[8] = [0.40, 0.28, 0.0]
        set_straight(9, 12, 0.50, length=0.28)
        pts[14] = [0.54, 0.46, 0.0]; pts[15] = [0.56, 0.37, 0.0]; pts[16] = [0.58, 0.28, 0.0]
        set_curled_fist(17, 20, 0.60)

    elif sign_name == "X":
        # Index finger hooked (bent 90 deg at PIP), other fingers curled into fist
        pts[3] = [0.46, 0.62, 0.02]; pts[4] = [0.49, 0.62, 0.03]
        set_hooked(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Y":
        # Thumb and pinky extended wide, middle 3 curled
        pts[3] = [0.38, 0.56, 0.0]; pts[4] = [0.30, 0.50, 0.0] # Thumb out
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        pts[18] = [0.60, 0.52, 0.0]; pts[19] = [0.63, 0.45, 0.0]; pts[20] = [0.67, 0.38, 0.0] # Pinky out

    elif sign_name == "Z":
        # Index finger extended straight, tracing Z
        pts[3] = [0.47, 0.62, 0.02]; pts[4] = [0.50, 0.62, 0.03]
        set_straight(5, 8, 0.45)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    # --- WLASL WORDS ---
    elif sign_name == "Hello":
        # Open 5-hand with fingers spread and tilted outward at temple
        pts[3] = [0.36, 0.56, 0.0]; pts[4] = [0.28, 0.50, 0.0]
        pts[6] = [0.42, 0.44, 0.0]; pts[7] = [0.40, 0.35, 0.0]; pts[8] = [0.38, 0.27, 0.0]
        pts[10] = [0.49, 0.43, 0.0]; pts[11] = [0.49, 0.34, 0.0]; pts[12] = [0.49, 0.25, 0.0]
        pts[14] = [0.55, 0.44, 0.0]; pts[15] = [0.57, 0.35, 0.0]; pts[16] = [0.59, 0.27, 0.0]
        pts[18] = [0.60, 0.48, 0.0]; pts[19] = [0.63, 0.42, 0.0]; pts[20] = [0.66, 0.36, 0.0]

    elif sign_name == "Stop":
        # Flat B/5 hand, fingers together straight up, palm vertical
        pts[3] = [0.42, 0.60, 0.0]; pts[4] = [0.38, 0.54, 0.0]
        set_straight(5, 8, 0.46)
        set_straight(9, 12, 0.50)
        set_straight(13, 16, 0.54)
        set_straight(17, 20, 0.58)

    elif sign_name == "Thank you":
        # Flat hand with thumb alongside
        pts[3] = [0.42, 0.55, 0.0]; pts[4] = [0.39, 0.48, 0.0]
        set_straight(5, 8, 0.45)
        set_straight(9, 12, 0.50)
        set_straight(13, 16, 0.54)
        set_straight(17, 20, 0.58)

    elif sign_name == "Please":
        # Flat 5 hand with palm facing chest (z inverted)
        pts[3] = [0.38, 0.58, -0.04]; pts[4] = [0.32, 0.52, -0.05]
        pts[6] = [0.44, 0.46, -0.03]; pts[7] = [0.44, 0.38, -0.04]; pts[8] = [0.44, 0.30, -0.05]
        pts[10] = [0.50, 0.44, -0.03]; pts[11] = [0.50, 0.36, -0.04]; pts[12] = [0.50, 0.28, -0.05]
        pts[14] = [0.54, 0.46, -0.03]; pts[15] = [0.54, 0.38, -0.04]; pts[16] = [0.54, 0.30, -0.05]
        pts[18] = [0.58, 0.50, -0.03]; pts[19] = [0.58, 0.43, -0.04]; pts[20] = [0.58, 0.36, -0.05]

    elif sign_name == "Sorry":
        # Fist rubbing chest (palm facing chest)
        pts[3] = [0.48, 0.59, -0.02]; pts[4] = [0.53, 0.59, -0.01]
        set_curled_fist(5, 8, 0.46, 0.02)
        set_curled_fist(9, 12, 0.50, 0.02)
        set_curled_fist(13, 16, 0.54, 0.02)
        set_curled_fist(17, 20, 0.58, 0.02)

    elif sign_name == "Yes":
        # S-fist upright (bobbing vertically)
        pts[3] = [0.48, 0.59, 0.05]; pts[4] = [0.53, 0.59, 0.06]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "No":
        # Index and middle snapping to thumb
        pts[3] = [0.46, 0.53, 0.02]; pts[4] = [0.48, 0.47, 0.03]
        pts[6] = [0.46, 0.47, 0.01]; pts[7] = [0.47, 0.46, 0.02]; pts[8] = [0.48, 0.47, 0.03]
        pts[10] = [0.49, 0.47, 0.01]; pts[11] = [0.49, 0.46, 0.02]; pts[12] = [0.48, 0.47, 0.03]
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Help":
        # A-fist with thumb straight up
        pts[3] = [0.42, 0.52, 0.02]; pts[4] = [0.42, 0.42, 0.04]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Good":
        # Clean thumbs-up
        pts[3] = [0.42, 0.52, 0.02]; pts[4] = [0.42, 0.40, 0.04]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Bad":
        # Flat hand with palm turned down
        pts[3] = [0.38, 0.58, 0.04]; pts[4] = [0.32, 0.52, 0.05]
        pts[6] = [0.44, 0.46, 0.06]; pts[7] = [0.44, 0.38, 0.08]; pts[8] = [0.44, 0.30, 0.10]
        pts[10] = [0.50, 0.44, 0.06]; pts[11] = [0.50, 0.36, 0.08]; pts[12] = [0.50, 0.28, 0.10]
        pts[14] = [0.54, 0.46, 0.06]; pts[15] = [0.54, 0.38, 0.08]; pts[16] = [0.54, 0.30, 0.10]
        pts[18] = [0.58, 0.50, 0.06]; pts[19] = [0.58, 0.43, 0.08]; pts[20] = [0.58, 0.36, 0.10]

    elif sign_name == "I love you":
        # Thumb, Index, Pinky extended; Middle and Ring curled
        pts[3] = [0.38, 0.56, 0.0]; pts[4] = [0.30, 0.50, 0.0]
        set_straight(5, 8, 0.45)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_straight(17, 20, 0.59)

    elif sign_name == "Want":
        # Curved claw hand palms facing up
        pts[3] = [0.40, 0.58, 0.04]; pts[4] = [0.37, 0.52, 0.06]
        set_curved_c(5, 8, 0.45)
        set_curved_c(9, 12, 0.50)
        set_curved_c(13, 16, 0.54)
        set_curved_c(17, 20, 0.58)

    elif sign_name == "Eat":
        # Flattened O-hand tilted toward mouth (negative Z depth)
        pts[3] = [0.46, 0.52, -0.04]; pts[4] = [0.49, 0.46, -0.05]
        pts[6] = [0.46, 0.46, -0.03]; pts[7] = [0.48, 0.45, -0.05]; pts[8] = [0.49, 0.46, -0.05]
        pts[10] = [0.49, 0.46, -0.03]; pts[11] = [0.49, 0.45, -0.05]; pts[12] = [0.49, 0.46, -0.05]
        pts[14] = [0.52, 0.48, -0.03]; pts[15] = [0.51, 0.47, -0.05]; pts[16] = [0.50, 0.47, -0.05]
        pts[18] = [0.55, 0.51, -0.03]; pts[19] = [0.53, 0.49, -0.05]; pts[20] = [0.50, 0.48, -0.05]

    elif sign_name == "More":
        # Flattened O-hands touching horizontally (positive Z depth)
        pts[3] = [0.46, 0.56, 0.03]; pts[4] = [0.49, 0.50, 0.04]
        pts[6] = [0.46, 0.49, 0.03]; pts[7] = [0.48, 0.49, 0.04]; pts[8] = [0.49, 0.50, 0.04]
        pts[10] = [0.49, 0.49, 0.03]; pts[11] = [0.49, 0.49, 0.04]; pts[12] = [0.49, 0.50, 0.04]
        pts[14] = [0.52, 0.51, 0.03]; pts[15] = [0.51, 0.51, 0.04]; pts[16] = [0.50, 0.51, 0.04]
        pts[18] = [0.55, 0.54, 0.03]; pts[19] = [0.53, 0.53, 0.04]; pts[20] = [0.50, 0.52, 0.04]

    elif sign_name == "Friend":
        # Hooked index finger (X-handshape)
        pts[3] = [0.46, 0.62, 0.02]; pts[4] = [0.49, 0.62, 0.03]
        set_hooked(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        set_curled_fist(17, 20, 0.58)

    elif sign_name == "Call me":
        # Y-handshape
        pts[3] = [0.38, 0.56, 0.0]; pts[4] = [0.30, 0.50, 0.0]
        set_curled_fist(5, 8, 0.46)
        set_curled_fist(9, 12, 0.50)
        set_curled_fist(13, 16, 0.54)
        pts[18] = [0.60, 0.52, 0.0]; pts[19] = [0.63, 0.45, 0.0]; pts[20] = [0.67, 0.38, 0.0]

    return pts

def extract_scale_invariant_features(pts):
    """
    Extracts 95 scale-invariant, translation-invariant geometric features:
    - Normalized landmark vectors (divided by palm scale)
    - 5 Fingertip-to-thumb distances
    - 5 Fingertip-to-wrist distances
    - 4 Adjacent fingertip distances (index-middle, middle-ring, ring-pinky, thumb-index)
    - 5 Joint angle cosines (MCP-PIP-TIP for all 5 digits)
    - 3D Palm normal vector
    - Hand aspect ratio
    """
    pts = np.asarray(pts, dtype=np.float32)
    wrist = pts[0]
    
    # Palm scale = distance from wrist to middle MCP
    palm_scale = float(np.linalg.norm(pts[9] - wrist))
    if palm_scale < 1e-4:
        palm_scale = 0.20

    # 1. Normalized landmark offsets: 21 * 3 = 63 features
    norm_pts = (pts - wrist) / palm_scale
    features = list(norm_pts.flatten())

    # 2. Pairwise fingertip to thumb tip distances: 4 features
    thumb_tip = pts[4]
    for tip_idx in [8, 12, 16, 20]:
        d = float(np.linalg.norm(pts[tip_idx] - thumb_tip)) / palm_scale
        features.append(d)

    # 3. Fingertip to wrist distances: 5 features
    for tip_idx in [4, 8, 12, 16, 20]:
        d = float(np.linalg.norm(pts[tip_idx] - wrist)) / palm_scale
        features.append(d)

    # 4. Adjacent finger separations: 4 features
    features.append(float(np.linalg.norm(pts[4] - pts[8])) / palm_scale)   # Thumb - Index
    features.append(float(np.linalg.norm(pts[8] - pts[12])) / palm_scale)  # Index - Middle
    features.append(float(np.linalg.norm(pts[12] - pts[16])) / palm_scale) # Middle - Ring
    features.append(float(np.linalg.norm(pts[16] - pts[20])) / palm_scale) # Ring - Pinky

    # 5. Joint angle cosines for each digit: 5 features
    for mcp, pip, tip in [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        v1 = pts[pip] - pts[mcp]
        v2 = pts[tip] - pts[pip]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-5 and n2 > 1e-5:
            cos_val = float(np.dot(v1, v2) / (n1 * n2))
        else:
            cos_val = 1.0
        features.append(cos_val)

    # 6. 3D Palm Normal: 3 features
    v_index = pts[5] - wrist
    v_pinky = pts[17] - wrist
    normal = np.cross(v_index, v_pinky)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 1e-5:
        normal = normal / norm_mag
    features.extend([float(normal[0]), float(normal[1]), float(normal[2])])

    # 7. Bounding box aspect ratio: 1 feature
    xs = pts[:, 0]
    ys = pts[:, 1]
    w = max(xs) - min(xs) + 1e-5
    h = max(ys) - min(ys) + 1e-5
    features.append(float(w / h))

    # 8. Thumb-to-knuckle relational offsets (A vs S vs T vs M vs N): 6 features
    features.append(float(thumb_tip[0] - pts[5, 0]) / palm_scale)
    features.append(float(thumb_tip[1] - pts[5, 1]) / palm_scale)
    features.append(float(thumb_tip[0] - pts[9, 0]) / palm_scale)
    features.append(float(thumb_tip[1] - pts[9, 1]) / palm_scale)
    features.append(float(thumb_tip[0] - pts[13, 0]) / palm_scale)
    features.append(float(thumb_tip[1] - pts[13, 1]) / palm_scale)

    # 9. Signed displacement of index tip relative to middle tip (R crossed vs U/V): 1 feature
    features.append(float(pts[8, 0] - pts[12, 0]) / palm_scale)

    # 10. Hand orientation direction vector (wrist to middle MCP: K/G vs P/Q): 3 features
    v_hand = (pts[9] - wrist) / palm_scale
    features.extend([float(v_hand[0]), float(v_hand[1]), float(v_hand[2])])

    # 11. Extension state indicators for all 5 digits: 5 features + 1 total count
    ext_indicators = []
    # Thumb: tip-to-wrist vs mcp-to-wrist
    thumb_ext = 1.0 if float(np.linalg.norm(pts[4] - wrist)) > float(np.linalg.norm(pts[2] - wrist)) * 1.15 else 0.0
    ext_indicators.append(thumb_ext)

    # 4 fingers: tip-to-wrist vs pip-to-wrist
    for t_idx, p_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        is_ext = 1.0 if float(np.linalg.norm(pts[t_idx] - wrist)) > float(np.linalg.norm(pts[p_idx] - wrist)) * 1.18 else 0.0
        ext_indicators.append(is_ext)

    features.extend(ext_indicators)
    features.append(float(sum(ext_indicators)))

    return features

def generate_augmented_dataset(samples_per_class=120):
    """
    Generates dataset with realistic 3D perturbations and bilateral symmetry.
    """
    X = []
    y = []

    print(f"Generating synthetic dataset for {len(ALL_CLASSES)} classes...")

    for sign in ALL_CLASSES:
        base = create_base_landmarks(sign)
        for _ in range(samples_per_class):
            pts = base.copy()

            # Random 3D scale variation (0.82 to 1.18)
            scale = np.random.uniform(0.82, 1.18)
            pts = (pts - pts[0]) * scale + pts[0]

            # Random 2D in-plane rotation (-18 to +18 deg)
            angle = np.random.uniform(-0.30, 0.30)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rel_x = pts[:, 0] - pts[0, 0]
            rel_y = pts[:, 1] - pts[0, 1]
            pts[:, 0] = pts[0, 0] + (rel_x * cos_a - rel_y * sin_a)
            pts[:, 1] = pts[0, 1] + (rel_x * sin_a + rel_y * cos_a)

            # Random 3D Gaussian landmark jitter
            pts[:, 0] += np.random.normal(0, 0.005, 21)
            pts[:, 1] += np.random.normal(0, 0.005, 21)
            pts[:, 2] += np.random.normal(0, 0.004, 21)

            # Bilateral mirror augmentation (50% left hand, 50% right hand)
            if np.random.rand() > 0.5:
                pts[:, 0] = 1.0 - pts[:, 0]

            feat = extract_scale_invariant_features(pts)
            X.append(feat)
            y.append(sign)

    # Incorporate captured samples from data.pickle if available
    try:
        with open("data.pickle", "rb") as f:
            real_data = pickle.load(f)
            real_x = real_data.get("data", [])
            real_y = real_data.get("labels", [])
            label_map = {"0": "A", "1": "B", "2": "L"}
            for feat_42, lbl in zip(real_x, real_y):
                target_sign = label_map.get(str(lbl))
                if target_sign and len(feat_42) == 42:
                    # Synthesize full 21x3 landmarks from 42 flat (x, y)
                    pts = np.zeros((21, 3), dtype=np.float32)
                    for idx in range(21):
                        pts[idx, 0] = feat_42[idx * 2]
                        pts[idx, 1] = feat_42[idx * 2 + 1]
                    feat = extract_scale_invariant_features(pts)
                    X.append(feat)
                    y.append(target_sign)
        print("Merged existing captured data from data.pickle successfully.")
    except Exception as e:
        print(f"Could not merge data.pickle: {e}")

    return np.asarray(X, dtype=np.float32), np.asarray(y)

def train_and_export():
    X, y = generate_augmented_dataset(samples_per_class=120)
    print(f"Dataset shape: {X.shape}, Labels count: {len(y)}")

    # Train Random Forest with single-threaded inference for instant execution
    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=26,
        min_samples_split=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1
    )

    print("Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    mean_acc = float(np.mean(cv_scores))
    print(f"Cross-Validation Accuracy: {mean_acc * 100:.2f}% (+/- {float(np.std(cv_scores)) * 100:.2f}%)")

    print("Fitting model on full dataset...")
    clf.fit(X, y)

    # Export unified model
    model_payload = {
        "model": clf,
        "classes": ALL_CLASSES,
        "alphabet_classes": ALPHABET_LEXICON,
        "wlasl_classes": WLASL_LEXICON,
        "feature_dim": X.shape[1],
        "accuracy": mean_acc,
        "version": "3.0.0-unified"
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Model saved to {MODEL_FILE}")

    # Also save to model.p for legacy scripts
    with open("model.p", "wb") as f:
        pickle.dump(model_payload, f)
    print("Model saved to model.p for backward compatibility.")

    # Update benchmark results
    benchmark = {
        "accuracy": round(mean_acc, 4),
        "total_classes": len(ALL_CLASSES),
        "alphabet_count": len(ALPHABET_LEXICON),
        "wlasl_count": len(WLASL_LEXICON),
        "feature_dim": X.shape[1],
        "version": "3.0.0-unified"
    }
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(benchmark, f, indent=2)
    print("Updated benchmark_results.json.")

if __name__ == "__main__":
    train_and_export()
