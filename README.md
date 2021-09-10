```bash
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
cd models
bash download_models.sh
cd ../
python3 predict.py data/dicom_00000001_000.dcm
```