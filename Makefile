.PHONY: build clean

build:
	cd atari_py/Arcade-Learning-Environment && \
	mkdir -p build && \
	cd build && \
	cmake .. && \
	make -j4

clean:
	rm -rf dist atari_py.egg-info
	cd atari_py/Arcade-Learning-Environment && \
	rm -rf build libale.so ale

package_data:
	( echo "Arcade-Learning-Environment/*.so" && echo "Arcade-Learning-Environment/ale" && cd atari_py && git ls-files |grep -v \\.py$ ) > atari_py/package_data.txt

upload:
	make clean
	rm -rf dist
	python setup.py sdist
	twine upload dist/*
