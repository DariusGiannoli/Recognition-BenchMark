import time 
import torch 
from ultralytics import YOLO 
from src.config import MODEL_PATHS
from .base import BaseDetector

class YOLODetector(BaseDetector): 
    
    def __init__(self, device = None): 
        
        self.device = device or (
                                "mps" if torch.backends.mps.is_available() else
                                "cuda" if torch.cuda.is_available() else 
                                "cpu"
                                )
        self.model_path = MODEL_PATHS['yolo']
        self.model = None
        self.load_model()
        
    def load_model(self): 
        
        try: 
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
        except Exception as e : 
            print("Error loading Yolo model", e)
            raise 
        
    def predict(self, image): 
        
        if self.model is None: 
            raise RuntimeError("Model not loaded. Call load_model() before predict().")
        
        #start clock
        t0= time.perf_counter()
        
        #inference 
        results = self.model(image, verbose = False, device = self.device, conf = 0.25)
        
        #stop clock 
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000
        
        #pars results 
        label = "background"
        confidence = 0.0
        
        if results[0].boxes: 
            top_box =  results[0].boxes[0]
            confidence = float(top_box.conf)
            class_id = int(top_box.cls)
            
            #Convert ID --> Name 
            label = self.model.names[class_id]
            
        return label, confidence, inference_time_ms