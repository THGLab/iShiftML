example_path="../../examples"
modelpath = 'iShiftML/models/TEV'
element="C"
mkdir local

for file in $example_path/*
do
name=`basename $file`
name="${name%.*}"
echo "start $name"

python ensemble_prediction.py -e $element --low_level_QM_file $example_path/$name.csv  --model_path $modelpath --output_folder local --include_low_level

echo "finished $name"
done
