import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Dataset Path
train_path = "../dataset"

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data
train = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# Validation Data
val = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train.class_indices)

# Load Base Model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Base Model (IMPORTANT for Speed)
for layer in base.layers:
    layer.trainable = False

# Add Custom Layers
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
output = Dense(train.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train,
    validation_data=val,
    epochs=5
)

# Save Model
model.save("plant_model.h5")