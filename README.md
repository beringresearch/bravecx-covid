# BraveCX-COVID (CovIx)
>> NOTE: Models and scripts are provided for research use only and are not for clinical and/or diagnostic use.

Full details of the algorithm and its validation are described in [Development and prospective validation of COVID-19 chest X-ray screening model for patients attending emergency departments](https://www.nature.com/articles/s41598-021-99986-3)

Chest X-Rays (CXRs) are the first-line investigation in patients presenting to emergency department (ED) with dyspnoea and are a valuable adjunct to clinical management of COVID-19 associated lung disease. Artificial Intelligence (AI) has the potential to facilitate rapid triage of CXRs for further patient testing and/or isolation. In this work we develop an AI algorithm, CovIx, to differentiate Normal, Abnormal, Non-COVID-19 Pneumonia, and COVID-19 CXRs using a multicentre cohort of 293,143 CXRs. The algorithm is prospectively validated in 3,289 CXRs acquired from patients presenting to ED with symptoms of COVID-19 across four sites in NHS Greater Glasgow and Clyde. CovIx achieves Area Under Receiver Operating Characteristics curve for COVID-19 of 0.86, with sensitivity and F1-score up to 0.83 and 0.71 respectively, and performs on-par with four board-certified radiologists. AI-based algorithms can identify CXRs with COVID-19 associated pneumonia, as well as distinguish non-COVID pneumonias in symptomatic patients presenting to ED.

## Installation

```bash
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
cd models
bash download_models.sh
cd ../
python3 predict.py data/dicom_00000001_000.dcm
```

## Model Inclusion Criteria

BraveCX-COVID classifiers were trained on X-Rays satisfying the following inclusion criteria:

* Body Part: Chest X-Ray
* Image Format: DICOM
* Minimum Resolution: 1500x1500
* View Position: Frontal (AP or PA)
* Minimum Bits Allocated: 16
* Patient Age: ≥16 years old

## Contact Us

* e-mail: info@beringresearch.com
* twitter: https://twitter.com/beringresearch