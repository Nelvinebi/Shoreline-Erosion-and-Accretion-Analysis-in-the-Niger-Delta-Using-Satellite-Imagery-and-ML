# Shoreline Erosion and Accretion Analysis in the Niger Delta Using Satellite Imagery and Machine Learning

## 📌 Project Overview
Coastal erosion and accretion are major environmental challenges in the **Niger Delta region of Nigeria**, driven by sea-level rise, wave action, sediment transport, and human activities such as oil and gas exploration.

This project demonstrates a **Machine Learning–based approach** for analyzing shoreline erosion and accretion using **synthetic satellite-derived data**. The workflow is designed to closely mimic real-world remote sensing applications and can be easily extended to real datasets such as **Landsat** or **Sentinel imagery**.

---

## 🎯 Objectives
- Simulate multi-temporal shoreline positions for the Niger Delta
- Identify shoreline **erosion**, **stability**, and **accretion** zones
- Train a Machine Learning model to classify shoreline change patterns
- Visualize shoreline dynamics spatially
- Provide a reproducible framework adaptable to real satellite data

---

## 🧠 Methodology
1. **Synthetic Data Generation**
   - Shoreline positions are simulated on a 2D grid representing satellite imagery.
   - Temporal shoreline movement is modeled to reflect erosion and accretion trends.

2. **Shoreline Change Detection**
   - Shoreline displacement between years (2005–2025) is computed.
   - Changes are classified as:
     - `-1` → Erosion  
     - `0` → Stable  
     - `1` → Accretion  

3. **Machine Learning Modeling**
   - A **Random Forest Classifier** is trained using shoreline position data.
   - The model predicts erosion and accretion zones.

4. **Visualization**
   - Shoreline positions
   - Observed erosion/accretion
   - ML-predicted shoreline change maps

---

## 📂 Project Structure
Shoreline-Erosion-Niger-Delta-ML/
│
├── shoreline_erosion_ml.py # Main Python script
├── Niger_Delta_Synthetic_Shoreline_Data.xlsx
│ ├── Shoreline_2005
│ ├── Shoreline_2025
│ └── Erosion_Accretion_Labels
├── README.md

yaml
Copy code

---

## 📊 Dataset Description
The project uses **synthetic data** to simulate satellite-derived shoreline positions.

### Excel File Contents
- **Shoreline_2005**: Simulated shoreline position values
- **Shoreline_2025**: Simulated shoreline position values
- **Erosion_Accretion_Labels**:
  - `-1` → Erosion
  - `0` → Stable
  - `1` → Accretion

> ⚠️ Synthetic data is used for demonstration and academic purposes only.

---

## 🛠️ Technologies Used
- Python  
- NumPy  
- Matplotlib  
- Scikit-learn  
- OpenPyXL  

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/shoreline-erosion-niger-delta-ml.git
Install dependencies:

bash
Copy code
pip install numpy matplotlib scikit-learn openpyxl
Run the script:

bash
Copy code
python shoreline_erosion_ml.py
📈 Sample Outputs
Synthetic shoreline position maps

Erosion and accretion classification maps

Machine Learning prediction results

Model evaluation metrics (confusion matrix, classification report)

🌍 Application to the Niger Delta
This framework is relevant for:

Coastal erosion monitoring

Environmental impact assessment

Climate change studies

Coastal zone management

Academic research and policy support in Nigeria

🔄 Future Improvements
Replace synthetic data with Landsat / Sentinel-2 imagery

Extract shorelines using NDWI or MNDWI

Integrate DSAS shoreline transects

Apply deep learning models (CNN, LSTM)

Incorporate sea-level rise and wave data

👤 Author
Ebingiye Nelvin Agbozu
Environmental Science | Machine Learning | Remote Sensing
Nigeria

📜 License
This project is for academic and research purposes.
Feel free to use, modify, and extend with proper attribution.
