#So First creating abstract base class 


#Importing packages 
from abc import ABC, abstractmethod 


#Now writing the class 

class BaseVLM(ABC):
    @abstractmethod 
    #Now creating analyze 

    def analyze(self, image_b64: str, system_prompt: str) -> str:
        ...
    @property
    @abstractmethod 

    def model_name(self) -> str: 
        ...
