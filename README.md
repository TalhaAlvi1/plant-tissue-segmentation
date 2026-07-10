# 🌱 Plant Tissue Segmentation

Computer vision pipeline that isolates plant leaves/stems from in-vitro test tube images — removes glass, background, and roots for clean, analysis-ready output.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Example

| Input | Segmented Output |
|---|---|
| <img width="1080" height="1033" alt="WhatsApp Image 2026-05-18 at 8 25 21 PM" src="https://github.com/user-attachments/assets/0e355901-9af0-4ce4-9cc1-70b09c555a97" /> | <img width="1080" height="1033" alt="2" src="https://github.com/user-attachments/assets/a133b42b-5448-40f6-8ad5-e51aa962cf37" /> |

| Input | Segmented Output |
|---|---|
|<img width="1080" height="1044" alt="WhatsApp Image 2026-05-18 at 8 25 29 PM" src="https://github.com/user-attachments/assets/6c687594-2805-4b5a-a606-3296dcc9b32e" />  |  <img width="1080" height="1044" alt="1" src="https://github.com/user-attachments/assets/54de515e-6ac9-467e-9689-1f97fc20b577" />  |


## Features

- 🧪 Auto-detects up to 8 test tubes per image (contour-based, falls back to equal-width slicing)
- 🍃 Extracts leaves + stems only, using combined HSV + LAB color segmentation
- 🚫 Excludes roots by masking the lower portion of each tube
- ✨ Strips glass reflections and bright/dark artifacts
- 🖼️ Outputs segmented plants on a clean black background, in original spatial position
- 📊 Side-by-side input/output comparison generator

## Pipeline

```mermaid
flowchart LR
    A[Input Image] --> B[Detect Test Tube Regions]
    B --> C["Color Segmentation<br/>HSV + LAB"]
    C --> D["Remove Roots<br/>bottom ~35-40%"]
    D --> E["Remove Glass Reflections<br/>& Small Noise"]
    E --> F["Composite on<br/>Black Background"]
    F --> G[Segmented Output]
```

## Project Structure

```
plant-tissue-segmentation/
├── plant_segmentation.py       # Baseline segmentation pipeline
├── improved_segmentation.py    # Enhanced accuracy (HSV + LAB, stricter filtering)
├── visualize_results.py        # Generates input/output comparison images
├── requirements.txt
├── input/                      # Place source images here
├── output/                     # Segmented results are written here
└── assets/                     # Example images used in this README
```

## Installation

```bash
git clone https://github.com/<your-username>/plant-tissue-segmentation.git
cd plant-tissue-segmentation
pip install -r requirements.txt
```

## Usage

```bash
# 1. Add images to input/
# 2. Run the pipeline
python plant_segmentation.py          # baseline
python improved_segmentation.py       # recommended, higher accuracy

# 3. (optional) Generate side-by-side comparisons
python visualize_results.py
```

Outputs are saved to `output/`, comparisons to `comparisons/`.

## Tuning

**Root cutoff** — controls how much of the tube's lower section is discarded:
```python
# improved_segmentation.py -> remove_roots_aggressive()
cutoff = int(height * 0.60)   # keep top 60%
```

**Green color range** — widen/narrow to capture different plant shades:
```python
lower1 = np.array([35, 40, 30])
upper1 = np.array([85, 255, 255])
```

## License

MIT
