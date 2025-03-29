
import numpy as np
import json
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import classification_report
import torch
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
import os

class Ensemble(object):
    def __init__(self):
                
        with open(os.getcwd()+"/processing/notes/finetuned_clinicalbert_logits.json", 'r') as f:
             self.nlp_logits = json.load(f)
             
        with open(os.getcwd()+"/processing/images/finetuned_resnet50_logits.json", 'r') as f:
            self.vision_logits = json.load(f)
        
        self.results = None

    def ensemble(self):

        # Normalize logits
        image_softmax_norm = F.softmax(torch.tensor(self.vision_logits), dim=1).numpy()
        notes_softmax_norm = F.softmax(torch.tensor(self.nlp_logits), dim=1).numpy()

        # Adjusted for Weighted F1 score for each class
        adjusted_f1_image = []
        adjusted_f1_notes = []

        for Benign, Malignant, Normal in image_softmax_norm:
            Benign *= 0.5085
            Malignant *= 0.6800
            Normal *= 0.7312
            adjusted_f1_image.append([Benign, Malignant, Normal])
        
        for Benign, Malignant, Normal in notes_softmax_norm:
            Benign *= 0.8857
            Malignant *= 0.7857
            Normal *= 0.955
            adjusted_f1_notes.append([Benign, Malignant, Normal])
        

        # Add score accross each class
        final_scores = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(adjusted_f1_image, adjusted_f1_notes)]
        final_scores = np.argmax(final_scores, axis=1)

        self.results = final_scores
        return
    
    def evaluate(self):
        val_list = ['P251_R_CM_MLO', 'P55_R_CM_MLO', 'P261_L_CM_MLO', 'P96_R_CM_MLO', 'P111_R_CM_MLO', 'P281_R_CM_MLO', 'P278_L_CM_MLO', 'P280_L_CM_MLO', 'P110_L_CM_MLO', 'P59_R_CM_MLO', 'P155_L_CM_MLO', 'P318_R_CM_MLO', 'P131_L_CM_MLO', 'P123_R_CM_MLO', 'P1_L_CM_MLO', 'P322_L_CM_MLO', 'P271_L_CM_MLO', 'P56_L_CM_MLO', 'P246_R_CM_MLO', 'P16_R_CM_MLO', 'P214_L_CM_MLO', 'P213_L_CM_MLO', 'P230_R_CM_MLO', 'P288_L_CM_MLO', 'P312_L_CM_MLO', 'P193_L_CM_MLO', 'P139_R_CM_MLO', 'P46_L_CM_MLO', 'P179_L_CM_MLO', 'P148_R_CM_MLO', 'P62_L_CM_MLO', 'P210_L_CM_MLO', 'P40_R_CM_MLO', 'P138_L_CM_MLO', 'P311_L_CM_MLO', 'P71_L_CM_MLO', 'P104_L_CM_MLO', 'P295_L_CM_MLO', 'P295_R_CM_MLO', 'P161_L_CM_MLO', 'P2_L_CM_MLO', 'P80_L_CM_MLO', 'P228_L_CM_MLO', 'P319_R_CM_MLO', 'P319_L_CM_MLO', 'P128_L_CM_MLO', 'P182_R_CM_MLO', 'P271_R_CM_MLO', 'P61_L_CM_MLO', 'P21_R_CM_MLO', 'P236_L_CM_MLO', 'P196_L_CM_MLO', 'P11_R_CM_MLO', 'P324_R_CM_MLO', 'P191_R_CM_MLO', 'P67_R_CM_MLO', 'P270_R_CM_MLO', 'P307_R_CM_MLO', 'P181_R_CM_MLO', 'P8_L_CM_MLO', 'P54_L_CM_MLO', 'P14_L_CM_MLO', 'P65_R_CM_MLO', 'P77_L_CM_MLO', 'P310_R_CM_MLO', 'P302_R_CM_MLO', 'P52_R_CM_MLO', 'P52_L_CM_MLO', 'P226_R_CM_MLO', 'P273_R_CM_MLO', 'P226_L_CM_MLO', 'P24_L_CM_MLO', 'P29_R_CM_MLO', 'P175_R_CM_MLO', 'P58_L_CM_MLO', 'P127_L_CM_MLO', 'P59_L_CM_MLO', 'P268_R_CM_MLO', 'P282_L_CM_MLO', 'P92_R_CM_MLO', 'P123_L_CM_MLO', 'P219_L_CM_MLO', 'P7_R_CM_MLO', 'P146_L_CM_MLO', 'P291_L_CM_MLO', 'P322_R_CM_MLO', 'P207_R_CM_MLO', 'P190_R_CM_MLO', 'P197_L_CM_MLO', 'P215_R_CM_MLO', 'P325_L_CM_MLO', 'P33_L_CM_MLO', 'P276_R_CM_MLO', 'P51_L_CM_MLO', 'P324_L_CM_MLO', 'P206_R_CM_MLO', 'P56_R_CM_MLO', 'P237_R_CM_MLO', 'P312_R_CM_MLO', 'P158_L_CM_MLO', 'P230_L_CM_MLO']

        clinical = read_csv(os.getcwd()+'/data/clinical_data.csv')
        
        label_encoder = LabelEncoder()
        clinical["Labels"] = label_encoder.fit_transform(clinical["Pathology Classification/ Follow up"])

        df_val = clinical[clinical['Image_name'].isin(val_list)]
        val_labels = list(df_val["Labels"])

        print("Accuracy: ", accuracy_score(val_labels, self.results))
        print("Confusion Matrix: ", confusion_matrix(val_labels, self.results))

        return



if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.ensemble()
    ensemble.evaluate()






