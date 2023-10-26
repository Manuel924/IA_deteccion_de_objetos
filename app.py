
# Estas líneas importan las bibliotecas necesarias para el servidor Flask, la detección de objetos con YOLOv3 
# y el procesamiento de imágenes con OpenCV.
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import requests

# URL de los archivos en GitHub para la detección de objetos con YOLOv3: 
# los modelos pre-entrenados, la configuración de la red neuronal y los nombres de las clases.
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Uso de esos modelos pre-entrenados que se descargan y guardan en local.
with open("yolov3.weights", "wb") as weights_file:
    weights_file.write(requests.get(weights_url).content)

with open("yolov3.cfg", "wb") as cfg_file:
    cfg_file.write(requests.get(cfg_url).content)

with open("coco.names", "wb") as coco_names_file:
    coco_names_file.write(requests.get(coco_names_url).content)

# Crea la instancia de flask para la visualizacion web
app = Flask(__name__)

# Cargar la configuración y los modelos pre-entrenados de la red neuronal de yolov3,
# contiene los valores y parametros que son aoprendidos de las imagenes etiquetadas
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Cargar las clases hace un llamado al archivo coco.names lo que hace que se realice la deteccion
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define una ruta para el servidor Flask que devuelve la página web
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = "uploaded_image.jpg"
        file.save(image_path)
        boxes, confidences, class_ids, detected_classes = detect_objects(image_path)

        # Aquí puedes procesar los resultados (contar personas, gatos y plantas.)
        # Por ahora, devolvemos los resultados en formato JSON
        results = [{'label': label, 'confidence': confidence} for label, confidence in zip(detected_classes, confidences)]

        return jsonify({'results': results})

# la ruta de la imagen a detectar y devuelve como resultado una lista de tuplas, 
# donde cada tupla contiene las coordenadas del cuadro delimitador del objeto detectado
def detect_objects(image_path):

    # leen la imagen desde la ruta especificada y almacenan su altura y ancho
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Preprocesar la imagen para la entrada de la red neuronal a un formato específico, escalarla a un 
    # tamaño determinado y normalizar sus píxeles. 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    #  listas vacías para almacenar la información sobre las detecciones
    class_ids = []
    confidences = []
    boxes = []

    # Bucle que recorre las salidas de la red neuronal y, para cada detección, almacena la información relevante en 
    # las listas creadas anteriormente. Si el rango de la detección es mayor que 0.5, 
    # se considera que la detección es válida y se añade a las listas.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Coordenadas del cuadro delimitador
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Esquina superior izquierda del cuadro delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supresión de no máximos (NMS)
    # Técnica que se utiliza para eliminar las detecciones redundantes.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Almacenar las detecciones finales
    final_boxes = [boxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]
    detected_classes = [classes[class_id] for class_id in final_class_ids]

    return final_boxes, final_confidences, final_class_ids, detected_classes

if __name__ == '__main__':
    app.run()