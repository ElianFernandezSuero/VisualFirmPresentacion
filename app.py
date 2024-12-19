from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

# Configuración de Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ELA_FOLDER = 'static/ela_images'
DATASET_FOLDER = 'dataset'
MODEL_WEIGHTS = 'deepfake_detector_weights.weights.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ELA_FOLDER'] = ELA_FOLDER

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ELA_FOLDER, exist_ok=True)

# -----------------------------------------------
# Función para realizar ELA
def perform_ela(image_path, quality=90, scale=30):
    original = Image.open(image_path).convert('RGB')
    temp_image_path = os.path.join(app.config['ELA_FOLDER'], 'temp_image.jpg')
    original.save(temp_image_path, 'JPEG', quality=quality)

    recompressed = Image.open(temp_image_path)
    ela_image = ImageChops.difference(original, recompressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale_factor = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)
    ela_image = ImageOps.autocontrast(ela_image)

    ela_output_path = os.path.join(app.config['ELA_FOLDER'], 'ela_result.jpg')
    ela_image.save(ela_output_path)
    return ela_output_path

# -----------------------------------------------
# Función para cargar dataset y entrenar el modelo
def train_model_with_dataset(dataset_dir, img_size=(224, 224), batch_size=32, epochs=10):
    # Generador de datos
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Modelo
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar modelo
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Guardar pesos
    model.save_weights(MODEL_WEIGHTS)
    print(f"Modelo entrenado y pesos guardados en {MODEL_WEIGHTS}")


    return model

# -----------------------------------------------
def extract_metadata(image_path):
    metadata = {}
    geo_data = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        for tag, value in tags.items():
            metadata[tag] = str(value)
            if "GPS" in tag:
                geo_data[tag] = str(value)
    except Exception as e:
        print(f"Error al extraer metadata:{e}")
        return metadata, geo_data
    
# Cargar el modelo preentrenado
def load_model():
    model = tf.keras.Sequential([
        tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights(MODEL_WEIGHTS)
    print("Modelo y pesos cargados correctamente.")
    return model

# Instanciar modelo
if os.path.exists(MODEL_WEIGHTS):
    deepfake_detector = load_model()
else:
    print("Pesos no encontrados. Entrenando el modelo...")
    deepfake_detector = train_model_with_dataset(DATASET_FOLDER)

# -----------------------------------------------
# Preprocesar la imagen para predicción
def preprocess_image(image_array, target_size=(224, 224)):
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array)
    else:
        image = image_array
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# -----------------------------------------------
# Rutas
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisis')
def analisis():
    return render_template('Analisis.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No se subió ninguna imagen."}), 400

    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    try:
        # Realizar ELA
        ela_image_path = perform_ela(image_path)

        # Extraer metadata y geo-tags
        metadata, geo_tags = extract_metadata(image_path)

        # Asegurarse de que metadata y geo_tags siempre tengan valores
        metadata = metadata if metadata else {"Info": "No metadata found"}
        geo_tags = geo_tags if geo_tags else {"Info": "No geo-tags found"}

        # Procesar la imagen ELA para predicción
        ela_image = Image.open(ela_image_path)
        ela_processed = preprocess_image(np.array(ela_image))

        # Realizar predicción
        prediction = deepfake_detector.predict(ela_processed)[0][0]
        deepfake_prob = round(prediction * 100, 2)
        authentic_prob = round((1 - prediction) * 100, 2)
        result = "Deepfake" if prediction >= 0.5 else "Auténtica"

        return render_template('result.html',
                               original_image=image_file.filename,
                               ela_image=os.path.basename(ela_image_path),
                               metadata=metadata,
                               geo_tags=geo_tags,
                               deepfake_prob=deepfake_prob,
                               authentic_prob=authentic_prob,
                               result=result)

    except Exception as e:
        return jsonify({"error": f"Error en el análisis: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ela_images/<filename>')
def ela_file(filename):
    return send_from_directory(app.config['ELA_FOLDER'], filename)


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)