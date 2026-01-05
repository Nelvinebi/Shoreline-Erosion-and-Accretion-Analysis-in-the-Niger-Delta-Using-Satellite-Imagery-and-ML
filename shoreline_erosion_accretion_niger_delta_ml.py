import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Step 1: Generate Synthetic Shoreline Data
# --------------------------------------------------

np.random.seed(42)

years = [1985, 1995, 2005, 2015, 2025]
grid_size = (100, 100)  # Simulated satellite image grid

def generate_shoreline(year, base_position=50):
    """
    Simulate shoreline position.
    Positive shift = accretion
    Negative shift = erosion
    """
    trend = (year - 1985) * np.random.uniform(-0.05, 0.05)
    shoreline = base_position + trend + np.random.normal(0, 2, grid_size)
    return shoreline

shorelines = {year: generate_shoreline(year) for year in years}

# --------------------------------------------------
# Step 2: Compute Shoreline Change (Erosion/Accretion)
# --------------------------------------------------

def shoreline_change(shoreline_t1, shoreline_t2):
    """
    Calculate shoreline movement
    """
    return shoreline_t2 - shoreline_t1

change_2005_2025 = shoreline_change(
    shorelines[2005],
    shorelines[2025]
)

# Labeling:
# Erosion = -1
# Stable = 0
# Accretion = +1
labels = np.where(
    change_2005_2025 < -1, -1,
    np.where(change_2005_2025 > 1, 1, 0)
)

# --------------------------------------------------
# Step 3: Prepare Machine Learning Dataset
# --------------------------------------------------

X = shorelines[2005].reshape(-1, 1)
y = labels.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 4: Train ML Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Step 5: Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Step 6: Predict Shoreline Change Map
# --------------------------------------------------

predicted_change = model.predict(
    shorelines[2005].reshape(-1, 1)
).reshape(grid_size)

# --------------------------------------------------
# Step 7: Visualization
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Synthetic Shoreline Position (2005)")
plt.imshow(shorelines[2005], cmap="terrain")
plt.colorbar(label="Shoreline Position")

plt.subplot(1, 3, 2)
plt.title("Observed Change (2005–2025)")
plt.imshow(labels, cmap="RdYlGn")
plt.colorbar(label="-1 Erosion | 0 Stable | 1 Accretion")

plt.subplot(1, 3, 3)
plt.title("Predicted Change (ML Output)")
plt.imshow(predicted_change, cmap="RdYlGn")
plt.colorbar(label="-1 Erosion | 0 Stable | 1 Accretion")

plt.tight_layout()
plt.show()
