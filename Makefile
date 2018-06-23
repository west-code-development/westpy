PYT=python

install: 
	${PYT} setup.py install --user --record files.txt

clean: 
	rm -rf westpy.egg-info
	rm -rf dist
	rm -rf build
	rm files.txt 
