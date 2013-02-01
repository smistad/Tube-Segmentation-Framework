#include "parameters.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include "SIPL/Exceptions.hpp"
using namespace std;

vector<string> split(string str, string delimiter) {
	vector<string> list;
	int start = 0;
	int end = str.find(delimiter);
	while(end != str.npos) {
		list.push_back(str.substr(start, end-start));
		start = end+1;
		end = str.find(delimiter, start);
	}
	// add last
	list.push_back(str.substr(start));

	return list;
}

paramList loadParameterPreset(paramList parameters) {
	// Check if parameters is set
    if(getParamStr(parameters, "parameters") != "none") {
    	std::string parameterFilename;
    	if(getParamStr(parameters, "centerline-method") == "gpu") {
    		parameterFilename = "parameters/centerline-gpu/" + getParamStr(parameters, "parameters");
    	} else if(getParamStr(parameters, "centerline-method") == "ridge") {
    		parameterFilename = "parameters/centerline-ridge/" + getParamStr(parameters, "parameters");
    	}
    	std::cout << parameterFilename << std::endl;
    	if(parameterFilename.size() > 0) {
    		// Load file and parse parameters
    		std::ifstream file(parameterFilename.c_str());
    		if(!file.is_open()) {
    			throw SIPL::IOException(parameterFilename.c_str(), __LINE__, __FILE__);
    		}

    		std::string line;
    		while(!file.eof()) {
				getline(file, line);
				if(line.size() == 0)
					continue;
    			// split string on the first space
    			int spacePos = line.find(" ");
    			if(spacePos != std::string::npos) {
    				// parameter with value
					std::string name = line.substr(0, spacePos);
					std::string value = line.substr(spacePos+1);
					parameters = setParameter(parameters, name, value);
    			} else {
    				// parameter with no value
    				parameters = setParameter(parameters, line, "true");
    			}
    		}
    		file.close();
    	}
    }
    return parameters;
}

paramList initParameters() {
	paramList parameters;

	std::ifstream file;
	std::string filename = "parameters/parameters";
	file.open(filename.c_str());
	if(!file.is_open())
		throw SIPL::IOException(filename.c_str(), __LINE__, __FILE__);
	string line;
	getline(file, line);
	getline(file, line); // throw away the first comment line
	while(file.good()) {
		int pos = 0;
		pos = line.find(" ");
		string name = line.substr(0, pos);
		line = line.substr(pos+1);
		pos = line.find(" ");
		string type = line.substr(0, pos);
		line = line.substr(pos+1);
		pos = line.find(" ");
		string defaultValue = line.substr(0,pos);

		if(type == "bool") {
			string description = line.substr(pos+2, line.length()-(pos+2)-1);
			BoolParameter v = BoolParameter(defaultValue == "true", description);
			parameters.bools[name] = v;
		} else if(type == "num") {
			line = line.substr(pos+1);
			pos = line.find(" ");
			float min = atof(line.substr(0,pos).c_str());
			line = line.substr(pos+1);
			pos = line.find(" ");
			float max = atof(line.substr(0,pos).c_str());
			line = line.substr(pos+1);
			float step = atof(line.c_str());

			int descriptionStart = line.find("\"");
			string description = line.substr(descriptionStart+1, line.length()-(descriptionStart+1)-1);
			NumericParameter v = NumericParameter(atof(defaultValue.c_str()), min, max, step, description);
			parameters.numerics[name] = v;
		} else if(type == "str") {
			vector<string> list ;
			int descriptionStart = line.find("\"");
			if(descriptionStart-pos > 1) {
				list = split(line.substr(pos+1, descriptionStart-(pos+1)), " ");
			}

			string description = line.substr(descriptionStart+1, line.length()-(descriptionStart+1)-1);
			StringParameter v = StringParameter(defaultValue, list, description);
			parameters.strings[name] = v;
		} else {
			throw exception();
		}

		getline(file, line);
	}

	return parameters;
}

paramList setParameter(paramList parameters, string name, string value) {
	if(parameters.bools.count(name) > 0) {
		BoolParameter v = parameters.bools[name];
		v.set(true);
		parameters.bools[name] = v;
	} else if(parameters.numerics.count(name) > 0) {
		NumericParameter v = parameters.numerics[name];
		if(!v.validate(atof(value.c_str()))) {
			cout << "invalid value for " << name << endl;
			throw exception();
		}
		v.set(atof(value.c_str()));
		parameters.numerics[name] = v;
	} else if(parameters.strings.count(name) > 0) {
		StringParameter v = parameters.strings[name];
		if(!v.validate(value)) {
			cout << "invalid value for " << name << endl;
			throw exception();
		}
		v.set(value);
		parameters.strings[name] = v;
	} else {
		throw exception();
	}

	return parameters;

}

float getParam(paramList parameters, string parameterName) {
	if(parameters.numerics.count(parameterName) == 0) {
		cout << parameterName << " not found" << endl;
 		throw exception();
	}
	NumericParameter v = parameters.numerics[parameterName];
	return v.get();
}

bool getParamBool(paramList parameters, string parameterName) {
	if(parameters.bools.count(parameterName) == 0) {
		cout << parameterName << " not found" << endl;
		throw exception();
	}
	BoolParameter v = parameters.bools[parameterName];
	return v.get();
}

string getParamStr(paramList parameters, string parameterName) {
	if(parameters.strings.count(parameterName) == 0) {
		cout << parameterName << " not found" << endl;
		throw exception();
	}
	StringParameter v = parameters.strings[parameterName];
	return v.get();
}

paramList getParameters(int argc, char ** argv) {
	paramList parameters = initParameters();

    // Go through each parameter, first parameter is filename
    // Try to see if the parameters parameter is set
    for(int i = 2; i < argc; i++) {
        string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            string nextToken;
            if(i+1 < argc) {
                nextToken = argv[i+1];
                if(nextToken.substr(0,2) == "--") {
                	nextToken = "";
                } else {
					i++;
                }
            } else {
            	nextToken = "";
            }
            if(token.substr(2) == "parameters")
				parameters = setParameter(parameters, token.substr(2), nextToken);
        }
    }

    // If a parameter preset is given load these values
    parameters = loadParameterPreset(parameters);

    // Go through each parameter, first parameter is filename
	for(int i = 2; i < argc; i++) {
        string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            string nextToken;
            if(i+1 < argc) {
                nextToken = argv[i+1];
                if(nextToken.substr(0,2) == "--") {
                	nextToken = "";
                } else {
					i++;
                }
            } else {
                nextToken = "";
            }
			parameters = setParameter(parameters, token.substr(2), nextToken);
        }
    }

	return parameters;
}

BoolParameter::BoolParameter(bool defaultValue, string description) {
	this->value = defaultValue;
	this->description = description;
}

bool BoolParameter::get() {
	return this->value;
}

void BoolParameter::set(bool value) {
	this->value = value;
}

NumericParameter::NumericParameter(float defaultValue, float min, float max, float step, string description) {
	this->value = defaultValue;
	this->min = min;
	this->max = max;
	this->step = step;
	this->description = description;
}

float NumericParameter::get() {
	return this->value;
}

void NumericParameter::set(float value) {
	if(this->validate(value)) {
		this->value = value;
	}
}

bool NumericParameter::validate(float value) {
	return value >= min && value <= max && floor((value-min)/step) == (value-min)/step;
}

StringParameter::StringParameter(string defaultValue, vector<string> possibilities, string description) {
	this->value = defaultValue;
	this->possibilities = possibilities;
	this->description = description;
}

string StringParameter::get() {
	return this->value;
}

void StringParameter::set(string value) {
	if(this->validate(value)) {
		this->value = value;
	}
}

bool StringParameter::validate(string value) {
	if(possibilities.size() > 0) {
		vector<string>::iterator it;
		bool found = false;
		for(it=possibilities.begin();it!=possibilities.end();it++){
			if(value == *it) {
				found = true;
				break;
			}
		}
		return found;
	} else {
		return true;
	}
}

float NumericParameter::getMax() const {
	return max;
}

void NumericParameter::setMax(float max) {
	this->max = max;
}

float NumericParameter::getMin() const {
	return min;
}

void NumericParameter::setMin(float min) {
	this->min = min;
}

float NumericParameter::getStep() const {
	return step;
}

void NumericParameter::setStep(float step) {
	this->step = step;
}

std::vector<std::string> StringParameter::getPossibilities() const {
	return possibilities;
}

std::string BoolParameter::getDescription() const {
	return description;
}

std::string NumericParameter::getDescription() const {
	return description;
}

std::string StringParameter::getDescription() const {
	return description;
}






