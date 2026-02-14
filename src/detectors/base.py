from abc import ABC, abstractmethod
import numpy as np 

class BaseDetector(ABC): 
    """
    The Interface (Blueprint).
    All models (YOLO, MobileNet, ResNet, RCE) must inherit from this class.
    
    This ensures that the benchmark script can treat them all exactly the same.
    """
    @abstractmethod
    def load_model(self): 
        """
        Initialize the model architecture and load weights from disk.
        This must happen before prediction.
        """
        pass 
    
    @abstractmethod
    def predict(self, image :np.ndarray): 
        """
        Run inference on a single image.
        
        Args:
            image (np.ndarray): A BGR image from OpenCV (Height, Width, Channels).
            
        Returns:
            tuple: A tuple containing exactly 3 elements:
                1. label (str): The name of the detected object (e.g., 'bird', 'mug').
                2. confidence (float): How sure the model is (0.0 to 1.0).
                3. inference_time (float): Processing time in milliseconds.
        """
        pass 
    
    