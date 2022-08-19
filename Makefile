PYT=python3

install:
	${PYT} setup.py install --user --prefix= --record files.txt

clean:
	rm -rf westpy.egg-info
	rm -rf dist
	rm -rf build
	rm -f files.txt
