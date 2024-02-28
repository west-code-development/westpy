PYT=python3

install:
	${PYT} -m pip install --user .

clean:
	rm -rf westpy.egg-info
	rm -rf dist
	rm -rf build
	rm -f files.txt
