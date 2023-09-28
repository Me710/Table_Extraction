from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import io

app = FastAPI()

# Initialize the feature extractor and model
feature_extractor = DetrFeatureExtractor()
model = TableTransformerForObjectDetection.from_pretrained("Christian710/table_detection_detr")

# Define COLORS for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

@app.post("/detect_table/")
async def detect_table(file: UploadFile):
    try:
        # Read the uploaded image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess the image
        encoding = feature_extractor(image, return_tensors="pt")

        # Perform a forward pass
        with torch.no_grad():
            outputs = model(**encoding)

        # Rescale bounding boxes
        width, height = image.size
        results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]

        # Visualize the results
        plot_results(image, results['scores'], results['labels'], results['boxes'])

        return JSONResponse(content={"message": "Table detection successful"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
