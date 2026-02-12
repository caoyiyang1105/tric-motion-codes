echo -e "Downloading T2M evaluators"

gdown --fuzzy https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8
rm -rf t2m

unzip t2m.zip
echo -e "Cleaning\n"
rm t2m.zip

echo -e "Downloading done!"
