import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

class TextClassificationModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.load_data()
        self.preprocess_data()
        self.prepare_data()
        self.build_model()

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)
        self.df['desc_process'] = self.df['desc_process'].fillna('')  # 将NaN值替换为空字符串
        self.df['desc_process'] = self.df['desc_process'].astype(str)  # 确保所有值都是字符串
        self.desc_process_cut = self.df['desc_process'].apply(lambda x: list(set(x.split())))
        self.labels = self.df['labels'].values

    def preprocess_data(self):
        self.desc_process_cut = [words if words else [""] for words in self.desc_process_cut]
        model_w2v = Word2Vec(self.desc_process_cut, vector_size=120, window=5, min_count=1)
        self.w2v_vectors = np.array([
            np.mean([model_w2v.wv[word] for word in words if word in model_w2v.wv], axis=0)
            if words else np.zeros(120)  # 用零向量替代空列表
            for words in self.desc_process_cut
        ])

    def prepare_data(self):
        x_train, x_val, y_train, y_val = train_test_split(self.w2v_vectors, self.labels, test_size=0.3, random_state=42)
        smote = SMOTE()
        self.x_train_resampled, self.y_train_resampled = smote.fit_resample(x_train, y_train)
        self.x_train = np.reshape(self.x_train_resampled, (self.x_train_resampled.shape[0], self.x_train_resampled.shape[1], 1))
        self.x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        self.y_val = y_val
        self.num_classes = len(np.unique(self.y_val))

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.x_train[0].shape))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
        self.history = self.model.fit(self.x_train, self.y_train_resampled, batch_size=128, epochs=30, validation_data=(self.x_val, self.y_val), callbacks=[early_stopping])

    def evaluate_model(self):
        y_pred = np.argmax(self.model.predict(self.x_val), axis=-1)
        print(classification_report(self.y_val, y_pred))
        acc = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, average='macro', zero_division=1)
        recall = recall_score(self.y_val, y_pred, average='macro', zero_division=1)
        f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=1)
        cm = confusion_matrix(self.y_val, y_pred)
        print('Accuracy:', acc)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1 Score:', f1)
        print('Confusion Matrix:\n', cm)

    def plot_roc_curve(self):
        y_prob = self.model.predict(self.x_val)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(self.y_val))[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = ['blue', 'green', 'red', 'yellow']  # 4个类别对应4种颜色
        for i, color in zip(range(self.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_val, np.argmax(self.model.predict(self.x_val), axis=-1))
        df_cm = pd.DataFrame(cm, index=[i for i in range(self.num_classes)], columns=[i for i in range(self.num_classes)])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def run(self):
        self.train_model()
        self.evaluate_model()
        self.plot_roc_curve()
        self.plot_confusion_matrix()


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    input_data = './data/selected_data_with_label.csv'
    model = TextClassificationModel(input_data)
    model.run()

