# EngineeringDiagram---Performance-Predictor
A pre-trained VLM model architecture aimed at identifying the optimal number of man hours, precision, machines required, dimensions of an X product mentioned by studying and analyzing the Engineering CAD diagrams provided. 



#Environment used: myenv 
To activate: source myenv/bin/activate 

#To have all the requirements:
pip install -r requirements.txt 

#After doing so, navigate to the following tests to run the evaluations


python tests/test_ingestion.py 
python tests/test_vlm.py 
python tests/test_alignment.py
python tests/test_manufacturing.py 

#To get the documents and activate the RAG pipeline: 

python knowledge_base/build_kb.py


Documents stored are: 
1) iso_tolerances.txt 
2) machinery_handbook.txt 
3) process_times.txt
4) surface_finish.txt 


Future work: 
Have to check the evaluation metrices 
Have to make the UI for the intiial upload 
Used dummy data as of now, need to check the confidentiality check 

