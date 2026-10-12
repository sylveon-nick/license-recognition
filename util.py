import os
import csv
import pandas as pd

def format_bbox(bbox):
    """Format bounding box coordinates cleanly as (x1, y1, x2, y2)."""
    return f"({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"

def save_detection_csv(detection_list, output_path):
    """
    Save structured detection records to a clean CSV file.
    
    Args:
        detection_list: List of dicts with keys:
            ['car_id', 'car_bbox', 'car_score', 'plate_bbox', 'plate_score', 'plate_text', 'text_score']
        output_path: Path to destination CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    for item in detection_list:
        rows.append({
            'Vehicle ID': item.get('car_id', 0),
            'Plate Number': item.get('plate_text', 'N/A'),
            'OCR Confidence': f"{item.get('text_score', 0) * 100:.1f}%",
            'Plate Confidence': f"{item.get('plate_score', 0) * 100:.1f}%",
            'Vehicle Confidence': f"{item.get('car_score', 0) * 100:.1f}%",
            'Plate Box (x1,y1,x2,y2)': format_bbox(item.get('plate_bbox', [0, 0, 0, 0])),
            'Vehicle Box (x1,y1,x2,y2)': format_bbox(item.get('car_bbox', [0, 0, 0, 0])),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
