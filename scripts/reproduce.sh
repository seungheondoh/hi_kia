# fold_case = ["M1","M2","M3" | ,"M6","F4", | "F5","F7","F8"]
# python3 preprocessing.py
python3 eval.py --cv_split M1 --gpus 0 --freeze_type feature
python3 eval.py --cv_split M2 --gpus 0 --freeze_type feature
python3 eval.py --cv_split M3 --gpus 0 --freeze_type feature
python3 eval.py --cv_split M4 --gpus 0 --freeze_type feature
python3 eval.py --cv_split F5 --gpus 0 --freeze_type feature
python3 eval.py --cv_split F6 --gpus 0 --freeze_type feature
python3 eval.py --cv_split F7 --gpus 0 --freeze_type feature
python3 eval.py --cv_split F8 --gpus 0 --freeze_type feature