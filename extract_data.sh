wget -r -N -c -np --user <CHANGE TO YOUR USER NAME> --ask-password https://physionet.org/files/mimiciii/1.4/
cd physionet.org/files/mimiciii/1.4
find . -name '*.gz' -exec gunzip '{}' \;
rm -rf *.gz
cd ..
mv 1.4 unzipped_files/
mkdir -p ../../../data
cp -R unzipped_files ../../../data/
