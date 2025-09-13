import cv2
import numpy as np
import pandas as pd
import random as rnd
import traceback
import tkinter as tk
from tkinter import ttk

# Load datasets
try:
    stimuli_df = pd.read_csv("../Clothes_colour_Perrett_Sprengelmeyer_Face_stimuli.csv")
    choices_df = pd.read_csv("../Clothes_colour_Participant_choices_Perrett_Sprengelmeyer_.csv")
except FileNotFoundError as e:
    print(f"Error: Could not load CSV file. Check paths: {e}")
    with open('recommendations.log', 'a') as f:
        f.write(f"Error: Could not load CSV file. Check paths: {e}\n")
    exit()

# Validate datasets
required_columns = {
    'stimuli_df': ['Face_skin_type', 'L', 'a', 'b'],
    'choices_df': ['tanned_fair_skin_face', 'Female_0_Male_1', 'R', 'G', 'B']
}
for df_name, cols in required_columns.items():
    df = locals()[df_name]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns in {df_name}: {missing}")
        with open('recommendations.log', 'a') as f:
            f.write(f"Error: Missing columns in {df_name}: {missing}\n")
        exit()

# Clean choices_df
choices_df = choices_df[choices_df['tanned_fair_skin_face'] != 'tanned_fair_skin_face']
choices_df = choices_df[choices_df['Female_0_Male_1'] != 'Female_0_Male_1']
for col in ['R', 'G', 'B', 'Female_0_Male_1']:
    choices_df[col] = pd.to_numeric(choices_df[col], errors='coerce')
choices_df = choices_df.dropna(subset=['tanned_fair_skin_face', 'Female_0_Male_1', 'R', 'G', 'B'])

# User inputs
gender_input = input("Enter gender (0 for female, 1 for male): ")
try:
    gender = int(gender_input)
    if gender not in [0, 1]:
        raise ValueError
except ValueError:
    print("Invalid gender input. Defaulting to 0 (female).")
    gender = 0

occasion = input("Enter occasion (e.g., casual, formal, party, wedding): ").strip().lower()
if not occasion:
    print("No occasion entered. Defaulting to 'casual'.")
    occasion = 'casual'

# Prompt for casual sub-context
sub_context = 'everyday'
if occasion == 'casual':
    sub_context = input("Enter casual context (e.g., summer, winter, everyday): ").strip().lower()
    if not sub_context:
        print("No sub-context entered. Defaulting to 'everyday'.")
        sub_context = 'everyday'

gender_str = 'women' if gender == 0 else 'men'

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    with open('recommendations.log', 'a') as f:
        f.write("Error: Could not open webcam.\n")
    exit()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar Cascade. Check OpenCV installation.")
    with open('recommendations.log', 'a') as f:
        f.write("Error: Could not load Haar Cascade. Check OpenCV installation.\n")
    cap.release()
    exit()

# Function to capture a single photo
def capture_photo():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            with open('recommendations.log', 'a') as f:
                f.write("Error: Failed to grab frame.\n")
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_region = None
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_region = frame[y:y+h, x:x+w]
        
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(faces) == 0:
                print("No face detected. Press 's' to try again, 'q' to quit.")
                continue
            if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                print("Face region too small. Press 's' to try again, 'q' to quit.")
                continue
            cv2.imwrite('captured_photo.jpg', face_region)
            cap.release()
            cv2.destroyAllWindows()
            return frame, face_region
        elif key == ord('q'):
            print("Exiting due to user quit.")
            cap.release()
            cv2.destroyAllWindows()
            return None, None

# Capture photo
print("Press 's' to capture a photo, 'q' to quit.")
frame, face_region = capture_photo()
if frame is None or face_region is None:
    print("Exiting due to no valid face capture.")
    with open('recommendations.log', 'a') as f:
        f.write("Exiting due to no valid face capture.\n")
    exit()

try:
    lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
    skin_lab = np.mean(lab, axis=(0, 1))
    print(f"Captured! Skin Lab values (mean): {skin_lab}")
except Exception as e:
    print(f"Error in skin tone detection: {e}")
    with open('recommendations.log', 'a') as f:
        f.write(f"Error in skin tone detection: {e}\n")
    exit()

def find_closest_skin(skin_lab, stimuli_df):
    try:
        distances = np.sqrt(((stimuli_df[['L', 'a', 'b']] - skin_lab[:3]) ** 2).sum(axis=1))
        return stimuli_df.iloc[distances.idxmin()]['Face_skin_type']
    except Exception as e:
        print(f"Error: Required columns (L, a, b) not found in stimuli_df: {e}")
        with open('recommendations.log', 'a') as f:
            f.write(f"Error: Required columns (L, a, b) not found in stimuli_df: {e}\n")
        return None

skin_type = find_closest_skin(skin_lab, stimuli_df)
if skin_type is None:
    print("Failed to match skin type. Check stimuli_df columns.")
    with open('recommendations.log', 'a') as f:
        f.write("Failed to match skin type. Check stimuli_df columns.\n")
    exit()
print(f"Matched skin type: {skin_type}")

try:
    pref_df = choices_df[(choices_df['tanned_fair_skin_face'] == skin_type) & 
                        (choices_df['Female_0_Male_1'] == gender)]
    print(f"Debug: pref_df size for skin_type={skin_type}, gender={gender}: {len(pref_df)}")
except Exception as e:
    print(f"Error filtering pref_df: {e}")
    with open('recommendations.log', 'a') as f:
        f.write(f"Error filtering pref_df: {e}\n")
    exit()

# Expanded preferred colors, excluding black for casual
preferred_colors = {
    'maroon': [128, 0, 0],
    'gold': [218, 165, 32],
    'cream': [245, 245, 220],
    'navy': [0, 0, 128],
    'ivory': [255, 255, 240],
    'white': [255, 255, 255],
    'emerald': [0, 128, 128],
    'coral': [255, 127, 127],
    'charcoal': [54, 69, 79],
    'lavender': [230, 230, 250]
} if occasion == 'wedding' else {
    'navy': [0, 0, 128],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'olive': [128, 128, 0],
    'maroon': [128, 0, 0],
    'teal': [0, 128, 128],
    'beige': [245, 245, 220],
    'denim': [30, 144, 255],
    'mustard': [255, 219, 88]
}

# Function to calculate RGB distance
def color_distance(rgb1, rgb2):
    return np.sqrt(sum((rgb1.iloc[i] - rgb2.iloc[i]) ** 2 for i in range(3))) if isinstance(rgb1, pd.Series) else np.sqrt(sum((rgb1[i] - rgb2[i]) ** 2 for i in range(3)))

def rgb_to_color_name(rgb):
    if isinstance(rgb, pd.Series):
        r, g, b = rgb.iloc[0], rgb.iloc[1], rgb.iloc[2]
    else:
        r, g, b = rgb[0], rgb[1], rgb[2]
    print(f"Debug: Converting RGB to color name - R: {r}, G: {g}, B: {b}")
    distances = {name: np.sqrt((r - pref_rgb[0])**2 + (g - pref_rgb[1])**2 + (b - pref_rgb[2])**2)
                 for name, pref_rgb in preferred_colors.items()}
    return min(distances, key=distances.get)

def generate_outfit():
    try:
        if not pref_df.empty:
            # Get unique colors from pref_df
            color_counts = pref_df.groupby(['R', 'G', 'B']).size().reset_index(name='count')
            print(f"Debug: Number of unique colors in pref_df: {len(color_counts)}")
            color_counts['min_distance'] = color_counts.apply(
                lambda row: min(color_distance(pd.Series([row['R'], row['G'], row['B']], index=['R', 'G', 'B']), pd.Series(pref_rgb, index=['R', 'G', 'B']))
                               for pref_rgb in preferred_colors.values()), axis=1)
            
            # Select top 5 colors
            top_colors = color_counts.sort_values(['min_distance', 'count'], ascending=[True, False]).head(5)
            print(f"Debug: Number of top colors selected: {len(top_colors)}")
            if len(top_colors) >= 2:
                # Pick upper color (outer garment)
                upper_color_row = top_colors.sample(n=1, weights='count', random_state=None)
                upper_rgb = upper_color_row.iloc[0][['R', 'G', 'B']].round().astype(int)
                # Filter out similar colors for lower body
                mask_lower = top_colors.apply(
                    lambda row: color_distance(pd.Series([row['R'], row['G'], row['B']], index=['R', 'G', 'B']), pd.Series(upper_rgb, index=['R', 'G', 'B'])) > 100, axis=1)
                remaining_colors_lower = top_colors[mask_lower]
                if not remaining_colors_lower.empty:
                    lower_color_row = remaining_colors_lower.sample(n=1, weights='count', random_state=None)
                    lower_rgb = lower_color_row.iloc[0][['R', 'G', 'B']].round().astype(int)
                else:
                    available_colors = [k for k in preferred_colors.keys() if color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(upper_rgb, index=['R', 'G', 'B'])) > 100]
                    lower_color_name = rnd.choice(available_colors) if available_colors else list(preferred_colors.keys())[0]
                    lower_rgb = pd.Series(preferred_colors[lower_color_name], index=['R', 'G', 'B'])
                # Pick inner shirt color for wedding
                inner_rgb = None
                if occasion == 'wedding':
                    mask_inner = top_colors.apply(
                        lambda row: color_distance(pd.Series([row['R'], row['G'], row['B']], index=['R', 'G', 'B']), pd.Series(upper_rgb, index=['R', 'G', 'B'])) > 100 and
                                    color_distance(pd.Series([row['R'], row['G'], row['B']], index=['R', 'G', 'B']), pd.Series(lower_rgb, index=['R', 'G', 'B'])) > 100, axis=1)
                    remaining_colors_inner = top_colors[mask_inner]
                    if not remaining_colors_inner.empty:
                        inner_color_row = remaining_colors_inner.sample(n=1, weights='count', random_state=None)
                        inner_rgb = inner_color_row.iloc[0][['R', 'G', 'B']].round().astype(int)
                    else:
                        available_colors = [k for k in preferred_colors.keys() if 
                                           color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(upper_rgb, index=['R', 'G', 'B'])) > 100 and
                                           color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(lower_rgb, index=['R', 'G', 'B'])) > 100]
                        inner_color_name = rnd.choice(available_colors) if available_colors else list(preferred_colors.keys())[0]
                        inner_rgb = pd.Series(preferred_colors[inner_color_name], index=['R', 'G', 'B'])
            else:
                upper_rgb = top_colors.iloc[0][['R', 'G', 'B']].round().astype(int)
                available_colors = [k for k in preferred_colors.keys() if k != 'white']
                lower_rgb = pd.Series(preferred_colors[rnd.choice(available_colors)], index=['R', 'G', 'B'])
                inner_rgb = None
                if occasion == 'wedding':
                    inner_rgb = pd.Series(preferred_colors[rnd.choice(available_colors)], index=['R', 'G', 'B'])
            print(f"Recommended Upper Body RGB: {upper_rgb}")
            print(f"Recommended Lower Body RGB: {lower_rgb}")
            if inner_rgb is not None:
                print(f"Recommended Inner Shirt RGB: {inner_rgb}")
        else:
            print("No valid color data for this skin type and gender. Using fallback colors.")
            available_colors = list(preferred_colors.keys())
            upper_color_name = rnd.choice(available_colors)
            available_colors_lower = [k for k in available_colors if color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(preferred_colors[upper_color_name], index=['R', 'G', 'B'])) > 100]
            lower_color_name = rnd.choice(available_colors_lower) if available_colors_lower else list(preferred_colors.keys())[0]
            upper_rgb = pd.Series(preferred_colors[upper_color_name], index=['R', 'G', 'B'])
            lower_rgb = pd.Series(preferred_colors[lower_color_name], index=['R', 'G', 'B'])
            inner_rgb = None
            if occasion == 'wedding':
                available_colors_inner = [k for k in available_colors if 
                                         color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(upper_rgb, index=['R', 'G', 'B'])) > 100 and
                                         color_distance(pd.Series(preferred_colors[k], index=['R', 'G', 'B']), pd.Series(lower_rgb, index=['R', 'G', 'B'])) > 100]
                inner_color_name = rnd.choice(available_colors_inner) if available_colors_inner else list(preferred_colors.keys())[0]
                inner_rgb = pd.Series(preferred_colors[inner_color_name], index=['R', 'G', 'B'])
            print(f"Fallback Upper Body RGB: {upper_rgb}")
            print(f"Fallback Lower Body RGB: {lower_rgb}")
            if inner_rgb is not None:
                print(f"Fallback Inner Shirt RGB: {inner_rgb}")
        
        upper_color = rgb_to_color_name(upper_rgb)
        lower_color = rgb_to_color_name(lower_rgb)
        inner_color = rgb_to_color_name(inner_rgb) if inner_rgb is not None else None
        
        # Expanded and refined clothing options
        occasion_map = {
            'casual': {
                'upper': ['T-shirt', 'Polo', 'Shirt', 'Hoodie', 'Sweater'] if gender == 0 else ['T-shirt', 'Polo', 'Shirt', 'Hoodie', 'Crewneck'],
                'lower': ['Jeans', 'Chinos', 'Joggers', 'Cargo Pants'] if gender == 0 else ['Jeans', 'Chinos', 'Joggers'] if sub_context != 'summer' else ['Jeans', 'Chinos', 'Joggers', 'Shorts']
            },
            'formal': {
                'upper': ['Formal Shirt', 'Blazer', 'Suit Jacket', 'Vest'],
                'lower': ['Trousers', 'Formal Pants', 'Slacks']
            },
            'party': {
                'upper': ['Kurta', 'Shirt', 'Jacket', 'Blazer', 'Tunic'] if gender == 0 else ['Shirt', 'Jacket', 'Blazer'],
                'lower': ['Tailored Fit Formal Pant', 'Trousers', 'Chinos', 'Slim Jeans']
            },
            'wedding': {
                'outer': ['Sherwani', 'Kurta', 'Suit Jacket', 'Bandhgala', 'Tuxedo'],
                'inner': ['Formal Shirt'],
                'lower': ['Churidar', 'Trousers', 'Dhoti', 'Formal Pants']
            }
        }
        
        # Randomly select clothing
        if occasion == 'wedding':
            upper_clothing = rnd.choice(occasion_map['wedding']['outer'])
            inner_clothing = rnd.choice(occasion_map['wedding']['inner'])
            lower_clothing = rnd.choice(occasion_map['wedding']['lower'])
        else:
            upper_clothing = rnd.choice(occasion_map.get(occasion, {}).get('upper', ['T-shirt']))
            inner_clothing = None
            lower_clothing = rnd.choice(occasion_map.get(occasion, {}).get('lower', ['Jeans']))
        
        return upper_rgb, lower_rgb, inner_rgb, upper_color, lower_color, inner_color, upper_clothing, lower_clothing, inner_clothing
    except Exception as e:
        print(f"Error in generate_outfit: {e}")
        print(traceback.format_exc())
        with open('recommendations.log', 'a') as f:
            f.write(f"Error in generate_outfit: {e}\n{traceback.format_exc()}\n")
        # Fallback to default values if generation fails
        upper_rgb = pd.Series(preferred_colors[rnd.choice(list(preferred_colors.keys()))], index=['R', 'G', 'B'])
        lower_rgb = pd.Series(preferred_colors[rnd.choice(list(preferred_colors.keys()))], index=['R', 'G', 'B'])
        inner_rgb = pd.Series(preferred_colors[rnd.choice(list(preferred_colors.keys()))], index=['R', 'G', 'B']) if occasion == 'wedding' else None
        upper_color = rgb_to_color_name(upper_rgb)
        lower_color = rgb_to_color_name(lower_rgb)
        inner_color = rgb_to_color_name(inner_rgb) if inner_rgb is not None else None
        if occasion == 'wedding':
            upper_clothing = rnd.choice(occasion_map['wedding']['outer'])
            inner_clothing = rnd.choice(occasion_map['wedding']['inner'])
            lower_clothing = rnd.choice(occasion_map['wedding']['lower'])
        else:
            upper_clothing = rnd.choice(occasion_map.get(occasion, {}).get('upper', ['T-shirt']))
            inner_clothing = None
            lower_clothing = rnd.choice(occasion_map.get(occasion, {}).get('lower', ['Jeans']))
        return upper_rgb, lower_rgb, inner_rgb, upper_color, lower_color, inner_color, upper_clothing, lower_clothing, inner_clothing

# GUI Setup
root = tk.Tk()
root.title("AI Stylist Outfit Catalog")
root.geometry("400x300")

# Catalog display
catalog_text = tk.Text(root, height=10, width=50)
catalog_text.pack(pady=10)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

def update_catalog():
    global outfit_catalog
    outfit_catalog = []
    # Add requested outfit first
    outfit_catalog.append({
        'index': 1,
        'upper_rgb': pd.Series([255, 255, 255], index=['R', 'G', 'B']),
        'lower_rgb': pd.Series([128, 0, 0], index=['R', 'G', 'B']),
        'inner_rgb': None,
        'upper_color': 'white',
        'lower_color': 'maroon',
        'inner_color': None,
        'upper_clothing': 'T-shirt',
        'lower_clothing': 'Jeans',
        'inner_clothing': None
    })
    # Generate one additional random outfit
    upper_rgb, lower_rgb, inner_rgb, upper_color, lower_color, inner_color, upper_clothing, lower_clothing, inner_clothing = generate_outfit()
    outfit_catalog.append({
        'index': 2,
        'upper_rgb': upper_rgb,
        'lower_rgb': lower_rgb,
        'inner_rgb': inner_rgb,
        'upper_color': upper_color,
        'lower_color': lower_color,
        'inner_color': inner_color,
        'upper_clothing': upper_clothing,
        'lower_clothing': lower_clothing,
        'inner_clothing': inner_clothing
    })

    catalog_text.delete(1.0, tk.END)
    catalog_text.insert(tk.END, "Outfit Catalog:\n==============\n")
    for outfit in outfit_catalog:
        catalog_text.insert(tk.END, f"Outfit {outfit['index']}:\n")
        catalog_text.insert(tk.END, f"  Upper body: {outfit['upper_color'].capitalize()} {outfit['upper_clothing']}\n")
        if outfit['inner_clothing']:
            catalog_text.insert(tk.END, f"  Inner shirt: {outfit['inner_color'].capitalize()} {outfit['inner_clothing']}\n")
        catalog_text.insert(tk.END, f"  Lower body: {outfit['lower_color'].capitalize()} {outfit['lower_clothing']}\n")
        catalog_text.insert(tk.END, f"  Upper RGB: {outfit['upper_rgb'].values}\n")
        catalog_text.insert(tk.END, f"  Lower RGB: {outfit['lower_rgb'].values}\n")
        if outfit['inner_rgb'] is not None:
            catalog_text.insert(tk.END, f"  Inner RGB: {outfit['inner_rgb'].values}\n")
        catalog_text.insert(tk.END, "\n")

btn_regenerate = tk.Button(btn_frame, text="Regenerate Catalog", command=update_catalog)
btn_regenerate.pack(side=tk.LEFT, padx=5)

btn_quit = tk.Button(btn_frame, text="Quit", command=root.quit)
btn_quit.pack(side=tk.LEFT, padx=5)

# Initial catalog generation
outfit_catalog = []
update_catalog()

root.mainloop()