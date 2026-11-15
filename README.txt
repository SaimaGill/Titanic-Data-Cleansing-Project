TITANIC DATA CLEANSING PROJECT

1. Download the Kaggle Titanic dataset:
   https://www.kaggle.com/competitions/titanic/data
2. Sign in to Kaggle if required and download the competition data.
3. Extract the downloaded ZIP file.
4. Copy train.csv into this project's Dataset folder.
5. Open Command Prompt/Terminal in this project folder.
6. Install packages:
       pip install -r requirements.txt
7. Run:
       python data_cleaning.py

OUTPUT:
- Output/cleaned_titanic.csv
- Output/cleaning_execution_report.txt

Important:
The script does not invent or delete unusual passenger records. Potential outliers are
flagged using the IQR rule so that valid extreme observations remain available for analysis.
