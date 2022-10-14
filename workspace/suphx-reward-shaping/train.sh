export result_path="result"
export data_path="resources/mjxproto"
export batch_size=256
export lr=0.001
export epochs=20


# suphx no logistic
python train.py $lr $epochs $batch_size --data_path=$data_path --result_path=$result_path --round_wise=0 --use_logistic=0 --use_saved_data=0

# suphx use logistic
python train.py $lr $epochs $batch_size  --data_path=$data_path --result_path=$result_path --round_wise=0 --use_logistic=1 --use_saved_data=1

# suphx use clip
python train.py $lr $epochs $batch_size  --data_path=$data_path --result_path=$result_path --round_wise=0 --use_logistic=0 --use_saved_data=1 --use_clip=1

# TD no logistic
for round in 7 6 5 4 3 2 1 0
do
python train.py $lr $epochs $batch_size --data_path=$data_path --result_path=$result_path --round_wise=1 --use_logistic=0 --target_round=$round --use_saved_data=0
done

# TD logistic
for round in 7 6 5 4 3 2 1 0
do
python train.py $lr $epochs $batch_size  --data_path=$data_path --result_path=$result_path --round_wise=1 --use_logistic=1 --target_round=$round --use_saved_data=1
done

for round in 7 6 5 4 3 2 1 0
do
python train.py $lr $epochs $batch_size --data_path=$data_path --result_path=$result_path --round_wise=1 --use_logistic=0 --use_clip=1 --target_round=$round --use_saved_data=0
done
