# Smarter-Mobility-Data-Challenge
**Author**\
Arthur Satouf

(Link to the competition)
https://codalab.lisn.upsaclay.fr/competitions/7192


## Getting started
Below is the structure and scripts used in the challange:

```bash
├── Arthur SATOUF.pdf
├── README.md
├── data_to_use
│   ├── data_ewm1.2.csv
│   ├── remCharEWM4.csv
│   ├── test.csv
│   ├── train.csv
│   └── train_onlyNext4EWM4andBack_EWM_remChar.csv
├── main.py
├── notebook
│   ├── area_catboost.ipynb
│   ├── cleaning.ipynb
│   ├── global_catboost.ipynb
│   ├── station_catboost.ipynb
│   └── visualization.ipynb
├── sample_result_submission
│   ├── area.csv
│   ├── global.csv
│   └── station.csv
└── sample_result_submission.zip
```
* **Arthur SATOUF.pdf** project report.
* **data_to_use** - data used to build the models and to forecast
    * **train_onlyNext4EWM4andBack_EWM_remChar.csv** data set input for Station.
    * **data_ewm1.2.csv** data set input for Area.
    * **remCharEWM4.csv** data set inout for Global.
    * **train.csv** (The basic) it been used as a source to create the above data sets and  to get validation index to imporove model for Station by using it as valaidation set.
    * **test.csv** to be forecasted and submitted.
* **main.py** to model and build sample_result_submission using CatBoost. 
* **notebook** a bunch of note-bookS to build each model and ***cleaning.ipynb*** is used to clean and prepoessing the data to build the input data sets **data_to_use**
* **sample_result_submission** my submission. 
* **sample_result_submission.zip** submission en ZIP.

# How to run the code 
- Clone the project.
- Open Terminal.
- Insert ```python main.py``` and wait until it finish to gererat data please note that it would take (15 to 20 min).

The result of the competition 

![image](https://github.com/arthur-75/Smarter-Mobility-Data-Challenge/assets/25811960/083577c5-b82e-4e8b-a7ca-d98a6499dc4e)

