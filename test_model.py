import io
import os
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'C:/Users/Huzaifa/PycharmProjects/DEFECT DETECTION/defect_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

model = load_model(model_path)

# Defect list corresponding to the model output
defects = [
    "Non-uniform color",
    "Tooling marks",
    "Exposed copper on the ends of the leads",
    "Bent or nonplanar leads",
    "Excessive or uneven plating",
    "Missing Pins",
    "Discoloration, dirt, or residues on the leads",
    "Scratches (or insertion marks) on the inside and outside faces of the leads",
    "Gross Oxidation",
    "Excessive solder on the leads",
    "Non-uniform thickness"
]

# Define exact coordinates for each checkbox (Yes and No)
checkbox_positions = [
    {"yes": (420, 485), "no": (450, 485)},  # Coordinates for "Non-uniform color"
    {"yes": (420, 472), "no": (450, 472)},  # Coordinates for "Tooling marks"
    {"yes": (420, 459), "no": (450, 459)},  # Coordinates for "Exposed copper on the ends of the leads"
    {"yes": (420, 445), "no": (450, 445)},  # Coordinates for "Bent or nonplanar leads"
    {"yes": (420, 433), "no": (450, 433)},  # Coordinates for "Excessive or uneven plating"
    {"yes": (420, 419), "no": (450, 419)},  # Coordinates for "Missing Pins"
    {"yes": (420, 406), "no": (450, 406)},  # Coordinates for "Discoloration, dirt, or residues on the leads"
    {"yes": (420, 385), "no": (450, 385)},  # Coordinates for "Scratches (or insertion marks)"
    {"yes": (420, 367), "no": (450, 367)},  # Coordinates for "Gross Oxidation"
    {"yes": (420, 354), "no": (450, 354)},  # Coordinates for "Excessive solder on the leads"
    {"yes": (420, 341), "no": (450, 341)}   # Coordinates for "Non-uniform thickness"
]

# Define coordinates for user input fields, including the width for centering
input_field_positions = {
    "Work Order": {"y": 764, "x_start": 37, "x_end": 231},
    "Customer": {"y": 764, "x_start": 240, "x_end": 403},
    "Part Number": {"y": 764, "x_start": 410, "x_end": 543},
    "Quantity": {"y": 743.1, "x_start": 37, "x_end": 231},
    "Manufacturer": {"y": 743.1, "x_start": 240, "x_end": 403},
    "Sample ID": {"y": 743.1, "x_start": 410, "x_end": 543}
}


def center_text(canvas_obj, text, x_start, x_end, y):
    """
    Centers text between x_start and x_end on the y-axis.
    """
    text_width = canvas_obj.stringWidth(text, "Helvetica", 8)  # Assuming font size of 10
    x_position = (x_start + x_end - text_width) / 2
    canvas_obj.drawString(x_position, y, text)


def predict_defects(image_folder_path):
    predictions = []
    target_size = (150, 150)  # Adjust this size based on your model's requirements

    for image_file in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, image_file)

        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize if required by your model

        # Predict using the model
        prediction = model.predict(img_array)[0]
        predictions.append(prediction)  # Append the full prediction vector for the image

    # Aggregate predictions for all defects
    avg_predictions = np.mean(predictions, axis=0)
    return (avg_predictions > 0.5).astype(int)  # Convert to binary output (0 or 1)


def update_pdf_with_predictions(input_pdf_path, output_pdf_path, predictions, user_inputs):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Create a new PDF with ReportLab to add the ticks, crosses, and user inputs
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont("Helvetica", 8)  # Set the font and size for user inputs

    # Add user input fields to the PDF, centered between the specified coordinates
    for field, value in user_inputs.items():
        coords = input_field_positions[field]
        center_text(can, value, coords["x_start"], coords["x_end"], coords["y"])

    # Iterate over predictions and use specific coordinates
    for i, prediction in enumerate(predictions):
        coords = checkbox_positions[i]
        if prediction == 1:
            # Draw a tick mark for "Yes"
            can.setFillColor(colors.green)
            can.drawString(*coords["yes"], "✔")
        else:
            # Draw a cross mark for "No"
            can.setFillColor(colors.red)
            can.drawString(*coords["no"], "✘")

    can.save()

    packet.seek(0)
    new_pdf = PdfReader(packet)
    existing_page = reader.pages[0]

    # Merge the new page with ticks/crosses and user inputs onto the original PDF
    existing_page.merge_page(new_pdf.pages[0])
    writer.add_page(existing_page)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_pdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the output to a new file
    with open(output_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)


# Example usage:
image_folder_path = 'D:/Image/temp'
input_pdf_path = 'D:/Optical Inspection - DIP Pkg.pdf'
output_pdf_path = 'C:/Users/Huzaifa/Desktop/Updated_Optical_Inspection_Report.pdf'

# Get predictions from the model
predictions = predict_defects(image_folder_path)

# Collect user inputs
user_inputs = {
    "Work Order": input("Enter Work Order: "),
    "Customer": input("Enter Customer: "),
    "Part Number": input("Enter Part Number: "),
    "Quantity": input("Enter Quantity: "),
    "Manufacturer": input("Enter Manufacturer: "),
    "Sample ID": input("Enter Sample ID: ")
}

# Update PDF with the model predictions and user inputs
update_pdf_with_predictions(input_pdf_path, output_pdf_path, predictions, user_inputs)