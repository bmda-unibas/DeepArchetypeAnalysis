conda_venv: environment.yml download_jaffe
	test -d venv || conda env create -f environment.yml

download_jaffe:
	test -d ./downloads || mkdir ./downloads
	wget 'https://zenodo.org/record/3451524/files/jaffedbase.zip?download=1' -O ./downloads/jaffedbase.zip
	unzip downloads/jaffedbase.zip -d ./downloads
	mv downloads/jaffedbase/* ./jaffe/images/
	#rename 's/^(.{2})\./$1-/' jaffe/images/*.tiff


	wget 'https://zenodo.org/record/3451524/files/README_FIRST.txt?download=1' -O ./downloads/jaffe_readme.txt
	sed -n '279,467p;468q' ./downloads/jaffe_readme.txt | sed '2d' | grep -v 'KM\-HA5' | \
	grep -v 'KM\-SA4' | grep -v 'KM\-DI2' | grep -v 'YN' | grep -v 'KR\-HA3' | grep -v 'NM\-DI2' | \
	grep -v 'TM\-HA4' > ./jaffe/labels.csv

clean:
	rm -rf ./downloads