


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

Current Issues:
1) InternVL2 is overfitting and memorizing on the grid pattern when generating the feature list and not giving the geometry correctly
2) InternVL2 cannot be finetuned because we lack training data, I will look through if i find any dataste from huggingface or kaggle
3) Best case would be shifting from InternVL2 to InternVL-2.5-8B or InternVL-26B or using Qwen2.5VL only for both annotations and feature list
4) After this, going to implement LLM as judge with Gemini

Current Progress:
1) Have changed the model from InternVL2 to Kimi-k2.6 with purchased API key from openrouter
2) Have used two VLA models Qwen 2.5 VL and Kimi-k2.6 and did the initial evaluation
3) Haved setup LLM evaluation using GPT and Gemini
4) Generates detailed manufacturing report on number of human hours required and the machine required
5) The results of LLM evaluation is saved to JSON file

