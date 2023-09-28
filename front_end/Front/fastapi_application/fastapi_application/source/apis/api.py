from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from ..services.detection import detection, structure_recognition,structure_recognition_2,visualize_results,save_plotted_image
from fastapi.responses import StreamingResponse
from ..utils.generic_utils import get_bytestream_from_image, image_input
from PIL import ImageDraw, Image
import io
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import torch

app = FastAPI()
# temp_directory = "/home/moetez/Table-recognition-internship/fastapi_application/temp/"
detected_boxes = []



@app.post("/table_detection/")
async def table_detection(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_tables = detection(decoded_image_rgb, [width, height], image_name)
    return detected_tables


@app.post("/result_visualisation_detection/")
async def result_visualisation_td(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_tables = detection(decoded_image_rgb, [width, height], image_name)
    image_bytestream = get_bytestream_from_image(decoded_image_rgb)
    image = Image.open(image_bytestream)
    draw = ImageDraw.Draw(image, "RGBA")
    for i in detected_tables:
        bbox = i.bbox
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline='red', width=3)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    result = output.getvalue()
    return StreamingResponse(io.BytesIO(result), media_type="image/png")

@app.post("/structure_recognition/")
async def table_structure(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_structure = structure_recognition(decoded_image_rgb, [width, height],image_name)
    return detected_structure

@app.post("/result_visualisation_structure/")
async def result_visualisation_ts(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_tables = structure_recognition(decoded_image_rgb, [width, height],image_name)
    image_bytestream = get_bytestream_from_image(decoded_image_rgb)
    image = Image.open(image_bytestream)
    draw = ImageDraw.Draw(image, "RGBA")
    for i in detected_tables:
        bbox = i.bbox
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline='red', width=3)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    result = output.getvalue()
    return StreamingResponse(io.BytesIO(result), media_type="image/png")

@app.post("/table_structure_visualisation/")
async def result_visualisation_ts(file: UploadFile = UploadFile(...), label_id: int = Query(..., description="Label ID to display 0: 'table' 1: 'table column', 2: 'table row', 3: 'table column header', 4: 'table projected row header',5: 'table spanning cell'")):
    processor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    try:
        image_content = await file.read()
        decoded_image_rgb = image_input(image_content)
        image_name = file.filename
        width, height, _ = decoded_image_rgb.shape

        # Detect objects in the image
        inputs = processor(images=decoded_image_rgb, return_tensors="pt")
        outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=[(height, width)])[0]

        # Filter results by the specified label_id
        label_ids = torch.tensor([label_id])
        filtered_results = {
            "scores": results["scores"][results["labels"] == label_id],
            "boxes": results["boxes"][results["labels"] == label_id],
            "labels": results["labels"][results["labels"] == label_id]
        }

        # Visualize results and convert the Matplotlib figure to an image
        plotted_boxes = visualize_results(decoded_image_rgb, filtered_results, label_ids)
        img_bytes = save_plotted_image(decoded_image_rgb, plotted_boxes)

        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


