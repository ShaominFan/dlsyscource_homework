# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	#c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
	c++ -g -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup -I/Library//Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -I/Users/fanshaomin/Library/Python/3.9/lib/python/site-packages/pybind11/include src/simple_ml_ext.cpp -o src/simple_ml_ext.so
