import cv2
import numpy as np
import pandas as pd
import dlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 自拍影像捕捉
def capture_selfie():
    """啟動攝影機，捕捉一張自拍照"""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ 無法開啟攝影機")
        return None

    print("📸 正在捕捉自拍照...")
    ret, frame = camera.read()
    if ret:
        selfie_path = 'selfie.jpg'
        cv2.imwrite(selfie_path, frame)
        print("✅ 自拍照片已保存為 selfie.jpg")
        cv2.imshow('Captured Selfie', frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print("❌ 無法捕捉影像")
        selfie_path = None

    camera.release()
    return selfie_path

# 性別判定
gender_model = tf.keras.models.load_model('gender_classification_model.h5')

def predict_gender(image_path):
    """根據自拍照預測性別"""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # 假設模型輸入大小為224x224
    image_normalized = np.expand_dims(image_resized, axis=0) / 255.0  # 正規化

    # 進行性別預測
    predictions = gender_model.predict(image_normalized)
    gender = "女" if predictions[0][0] < 0.5 else "男"
    
    print(f"預測性別: {gender}")
    return gender

# 臉型分析
def analyze_face_shape(image_path):
    """分析臉型"""
    print("\n🤖 正在進行臉型分析...")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = detector(gray)
    if len(faces) == 0:
        print("❌ 未偵測到人臉，無法進行臉型分析。")
        return None

    for face in faces:
        landmarks = predictor(gray, face)
        jawline = [landmarks.part(i) for i in range(0, 17)]
        width = jawline[-1].x - jawline[0].x
        height = face.bottom() - face.top()

        aspect_ratio = height / width
        if aspect_ratio > 1.5:
            return "長型臉"
        elif aspect_ratio > 1.2:
            return "橢圓形臉"
        elif aspect_ratio > 1.0:
            return "圓形臉"
        else:
            return "方形臉"

# 訓練 MobileNet/ResNet 模型
def train_hairstyle_model():
    """訓練髮型分類模型"""
    print("\n🧠 使用 MobileNetV2 訓練髮型分類模型...")
    df = pd.read_csv('hairstyle_data.csv')
    image_paths = df['Image_Path'].values
    labels = df['Hairstyle_Name'].values

    features = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img_to_array(img))
            features.append(img)
    
    X = np.array(features)
    y = LabelEncoder().fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(np.unique(y)), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"✅ 模型準確度: {accuracy:.2f}")
    
    model.save('hairstyle_model.h5')  # 儲存訓練好的模型
    return model

# 推薦髮型
def recommend_hairstyle(face_shape, gender, model):
    """推薦適合的髮型"""
    print(f"\n🎯 根據臉型『{face_shape}』及性別『{gender}』推薦適合的髮型...")

    df = pd.read_csv('hairstyle_data.csv')
    filtered_df = df[df['Face_Shape'] == face_shape]
    
    # 根據性別篩選髮型
    filtered_df = filtered_df[filtered_df['Gender'] == gender]

    sample_img = cv2.imread(filtered_df['Image_Path'].iloc[0])
    sample_img = cv2.resize(sample_img, (224, 224))
    sample_img = preprocess_input(img_to_array(sample_img))
    sample_img = np.expand_dims(sample_img, axis=0)
    
    prediction = model.predict(sample_img)
    predicted_label = np.argmax(prediction)
    hairstyle_name = df['Hairstyle_Name'].iloc[predicted_label]
    
    print(f"✅ 推薦髮型：{hairstyle_name}")
    return filtered_df['Image_Path'].iloc[predicted_label]

# 髮型套用與對比輸出
def apply_hairstyle(selfie_path, hairstyle_path):
    """將推薦髮型套用到自拍照"""
    selfie = cv2.imread(selfie_path)
    hairstyle = cv2.imread(hairstyle_path)
    hairstyle = cv2.resize(hairstyle, (selfie.shape[1], selfie.shape[0]))

    # 在對比圖中加上標註
    combined = np.hstack((selfie, hairstyle))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Before', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, 'After', (selfie.shape[1] + 10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    result_path = 'result_comparison.jpg'
    cv2.imwrite(result_path, combined)
    print(f"✅ 對比圖已保存為 {result_path}")
    cv2.imshow('Before and After', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()