# fold_case = ["M1","M2","M3" | ,"M6","F4", | "F5","F7","F8"]
python3 preprocessing.py
python3 train.py --cv_split $1 --gpus 0 --freeze_type $2
python3 eval.py --cv_split $1 --gpus 0 --freeze_type $2