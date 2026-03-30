import cv2
import numpy as np
from skimage.metrics import structural_similarity
import os

def compute_ratio(base_path, compare_path):
    base = cv2.imread(base_path)
    comp = cv2.imread(compare_path)
    if base is None or comp is None:
        return None, "Failed to load"

    h, w = base.shape[:2]
    comp_resized = cv2.resize(comp, (w, h))
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    comp_gray = cv2.cvtColor(comp_resized, cv2.COLOR_BGR2GRAY)

    # ORB alignment
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(base_gray, None)
    kp2, des2 = orb.detectAndCompute(comp_gray, None)
    
    good_matches = 0
    warped = comp_gray
    
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        good_matches = len(good)
        
        if good_matches >= 10:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
            if M is not None:
                warped = cv2.warpPerspective(comp_gray, M, (w, h))
                good_matches = int(mask.sum())

    # === METRIC 1: Match score (saturates at 50 matches) ===
    match_score = min(good_matches / 50.0, 1.0)  # 50+ matches = fully matched
    match_diff = 1.0 - match_score  # 0 = same, 1 = different

    # === METRIC 2: Blurred SSIM on aligned images ===
    my, mx = int(h * 0.15), int(w * 0.15)
    base_roi = cv2.GaussianBlur(base_gray[my:h-my, mx:w-mx], (31, 31), 0)
    comp_roi = cv2.GaussianBlur(warped[my:h-my, mx:w-mx], (31, 31), 0)
    ssim_val, _ = structural_similarity(base_roi, comp_roi, full=True)
    ssim_change = 1.0 - ssim_val

    # === METRIC 3: Histogram (Bhattacharyya on HSV) ===
    base_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    comp_hsv = cv2.cvtColor(comp_resized, cv2.COLOR_BGR2HSV)
    hist_base = cv2.calcHist([base_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_comp = cv2.calcHist([comp_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_base, hist_base)
    cv2.normalize(hist_comp, hist_comp)
    bhatt = cv2.compareHist(hist_base, hist_comp, cv2.HISTCMP_BHATTACHARYYA)

    # === Combined: match_diff and blur_ssim are the strongest signals ===
    ratio = (
        0.40 * match_diff       # feature match (best same/different signal)
        + 0.35 * ssim_change    # blurred SSIM after alignment
        + 0.25 * bhatt          # color distribution
    )
    ratio = round(min(max(ratio, 0.0), 1.0), 4)

    return ratio, {
        "matches": good_matches,
        "match_diff": round(match_diff, 4),
        "blur_ssim": round(ssim_change, 4),
        "bhatt": round(bhatt, 4),
    }

test_dir = "/Users/solyshawkat/Desktop/Ai Computer Vision Model/new test data"
base = os.path.join(test_dir, "base.jpg")
images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg') and f != 'base.jpg'])

print("=" * 70)
print("TEST v6 — Match Score (saturating) + Blurred SSIM + Bhattacharyya")
print("=" * 70)
for img_name in images:
    img_path = os.path.join(test_dir, img_name)
    print(f"\nbase.jpg vs {img_name}:")
    ratio, details = compute_ratio(base, img_path)
    if ratio is not None:
        print(f"    {details}")
        print(f"    >>> RATIO: {ratio * 100:.1f}%")
print("\n" + "=" * 70)

# Also test base vs base (should be 0%)
print(f"\nbase.jpg vs base.jpg (SELF):")
ratio, details = compute_ratio(base, base)
print(f"    {details}")
print(f"    >>> RATIO: {ratio * 100:.1f}%")
