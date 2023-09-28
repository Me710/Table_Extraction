from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from ..datamodel.output import Output
import numpy as np
from typing import List
from ..utils.generic_utils import crop_object
import matplotlib.pyplot as plt
import io


# Define COCO labels
id2label = {
    0: 'table',
    1: 'table column',
    2: 'table row',
    3: 'table column header',
    4: 'table projected row header',
    5: 'table spanning cell'
}

# Define colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def detection(image: np.ndarray, image_size: List[int], image_name: str):
    processor = AutoFeatureExtractor.from_pretrained("Christian710/table_detection_detr")
    model = AutoModelForObjectDetection.from_pretrained("Christian710/table_detection_detr")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=[(image_size[0], image_size[1])])[
        0
    ]
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result = Output(
                image_name=image_name,
                bbox=box
        )
        boxes.append(result)
    return boxes

def structure_recognition(image: np.ndarray, image_size: List[int],image_name:str):
    processor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    #detected_tables = detection(image, image_size, image_name)
    #print(detected_tables)
    # Assuming detected_tables is a list of Output objects
    #bbox_list = [output.bbox for output in detected_tables]
    #print(bbox_list)
    #image_crop = crop_object(image, bbox_list)
    #image = image_crop
    #width, height = image.size
    #image.resize((int(width * 0.5), int(height * 0.5)))
    #target_sizes = [image.size[::-1]]
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=[(image_size[0], image_size[1])])[0]
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result = Output(
                image_name=image_name,
                bbox=box
        )
        boxes.append(result)
    return boxes


def structure_recognition_2(image: np.ndarray, image_size: List[int], image_name: str):
    processor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    detected_tables = detection(image, image_size, image_name)
    print(detected_tables)
    # Assuming detected_tables is a list of Output objects
    bbox_list = [output.bbox for output in detected_tables]
    #print(bbox_list)
    image_crop = crop_object(image, bbox_list)
    #image = image_crop
    #width, height = image.size
    #image.resize((int(width * 0.5), int(height * 0.5)))
    #target_sizes = [image.size[::-1]]
    '''inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=[(image_size[0], image_size[1])])[0]
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result = Output(
                image_name=image_name,
                bbox=box
        )
        boxes.append(result)'''
    return image_crop

def visualize_results(image, results, label_ids):
    plotted_boxes = []  # List to store the plotted bounding boxes
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100

    for score, (xmin, ymin, xmax, ymax), label_id, c in zip(results["scores"].tolist(), results["boxes"].tolist(),
                                                           results["labels"].tolist(), colors):
        if label_id in label_ids:
            label = id2label.get(label_id, 'Unknown Label')
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            plotted_boxes.append((xmin, ymin, xmax, ymax))

    plt.axis('off')
    #plt.show()
    return plotted_boxes

def save_plotted_image(image, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100

    for (xmin, ymin, xmax, ymax), c in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))

    plt.axis('off')
    
    # Save the Matplotlib figure to a BytesIO buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_bytes = img_buffer.getvalue()
    plt.close()

    return img_bytes