# EconAI_Lab

Setting
- 1. clone this repo and upload all of it onto your google drive; local projects
- 2. use google colab pro
- 3. mount your google colab on your google drive; drive.mount('/content/drive') --> mkdir kdd --> cd kdd
- 4. run
- 5. if you made any important progress, please push your local projects on this repo

KDD Algorithm 
- Procedure (80/100) as of 26 July
  - 1) Meta_Data.py 
  - 2) pretrain.py
  - 3) extract_cnr.py
  - 4) vanilla_deepcluster.py
  - 5) extract_vanilla_cluster.py
  - 6) siScore.py
  - 7) extract_score.py

Use Local Runtime
- pip install jupyter_http_over_ws
- jupyter serverextension enable --py jupyter_http_over_ws
- cd (your work directory)
- jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
