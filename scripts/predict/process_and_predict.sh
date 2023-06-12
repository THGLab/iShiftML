example_path="../../examples"
element="H"
mkdir local/$element

for file in $example_path/xyzs/*
do
name=`basename $file`
name="${name%.*}"
echo "start $name"

python prepare_data.py $example_path/low_level_df/$name.csv --xyz_file $example_path/xyzs/$name.xyz --high_level_QM_calculation $example_path/high_level_df/$name.csv
python ensemble_prediction.py $element --has_target --include_low_level
mv local/ensemble_prediction_${element}_test.csv local/$element/$name.csv

echo "finished $name"
done
