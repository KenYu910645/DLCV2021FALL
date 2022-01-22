cd data
wget https://zenodo.org/record/3723295/files/subset0.zip?download=1 -O subset0.zip
wget https://zenodo.org/record/3723295/files/subset1.zip?download=1 -O subset1.zip
wget https://zenodo.org/record/3723295/files/subset2.zip?download=1 -O subset2.zip
wget https://zenodo.org/record/3723295/files/subset3.zip?download=1 -O subset3.zip
zip -FF subset0.zip -O subset0.fixed.zip && unzip subset0.fixed.zip
zip -FF subset1.zip -O subset1.fixed.zip && unzip subset1.fixed.zip
zip -FF subset2.zip -O subset2.fixed.zip && unzip subset2.fixed.zip
zip -FF subset3.zip -O subset3.fixed.zip && unzip subset3.fixed.zip
mkdir train
mkdir test
mv subset0/* test/
mv subset*/* train/
rm *.zip
rm -r -f subset*
cd ..
