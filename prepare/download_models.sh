rm -rf save
mkdir save
cd save

echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1f124-Ochdeh0Au1-sXF7TeimjIMaw9R_/view?usp=sharing

echo -e "Unzipping tric_motion_model_humanml.zip"
unzip tric_motion_model_humanml.zip

echo -e "Cleaning tric_motion_model_humanml.zip"
rm tric_motion_model_humanml.zip
