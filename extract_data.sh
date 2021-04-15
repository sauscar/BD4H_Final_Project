cd physionet.org/files/mimiciii/1.4
find . -name '*.gz' -exec gunzip '{}' \;
rm -rf *.gz
cd ..
mv 1.4 unzipped_files/
mkdir -p ../../../data
cp -R unzipped_files ../../../data/
