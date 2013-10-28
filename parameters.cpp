#include "parameters.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include "SIPL/Exceptions.hpp"
#include "tsf-config.h"
#include <locale>
#include <sstream>
#include "tsf-config.h"
using namespace std;

float stringToFloat(string str) {
	float value = 0.0f;
	istringstream istr(str);

	istr.imbue(locale("C"));
	istr >> value;
	return value;
}

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

void printAllParameters() {
    paramList parameters = initParameters(std::string(PARAMETERS_DIR));
	unordered_map<std::string, BoolParameter>::iterator bIt;
	unordered_map<std::string, NumericParameter>::iterator nIt;
	unordered_map<std::string, StringParameter>::iterator sIt;

    printf("%25s \t Default \t\t Choices \t\t Description\n", "Name");
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
	for(bIt = parameters.bools.begin(); bIt != parameters.bools.end(); ++bIt){
		printf("%25s \t %s \t\t true/false \t\t %s \n", bIt->first.c_str(), bIt->second.get() == 0 ? "false":"true", bIt->second.getDescription().c_str());
	}

	for(nIt = parameters.numerics.begin(); nIt != parameters.numerics.end(); ++nIt){
        if(nIt->second.getStep() >= 1.0f) {
            printf("%25s \t %.0f \t\t %.0f-%.0f \t\t %s \n", nIt->first.c_str(), nIt->second.get(), nIt->second.getMin(), nIt->second.getMax(), nIt->second.getDescription().c_str());
        } else {
            printf("%25s \t %.3f \t\t %.3f-%.3f \t\t %s \n", nIt->first.c_str(), nIt->second.get(), nIt->second.getMin(), nIt->second.getMax(), nIt->second.getDescription().c_str());
        }
	}
	for(sIt = parameters.strings.begin(); sIt != parameters.strings.end(); ++sIt){
        std::string possibilitiesString = "";
        std::vector<std::string> possibilities = sIt->second.getPossibilities();
        std::vector<std::string>::iterator it;
        for(it = possibilities.begin(); it != possibilities.end(); ++it) {
            possibilitiesString += *it;
            possibilitiesString += " ";
        }
        printf("%25s \t %s \t\t %s \t\t %s\n", sIt->first.c_str(), sIt->second.get().c_str(), possibilitiesString.c_str(), sIt->second.getDescription().c_str());
	}
}

void loadParameterPreset(paramList &parameters, std::string parameter_dir) {
	// Check if parameters is set
    if(getParamStr(parameters, "parameters") != "none") {
    	std::string parameterFilename;
    	if(getParamStr(parameters, "centerline-method") == "gpu") {
    		parameterFilename = parameter_dir+"/centerline-gpu/" + getParamStr(parameters, "parameters");
    	} else if(getParamStr(parameters, "centerline-method") == "test") {
    		parameterFilename = parameter_dir+"/centerline-test/" + getParamStr(parameters, "parameters");
    	} else if(getParamStr(parameters, "centerline-method") == "ridge") {
    		parameterFilename = parameter_dir+"/centerline-ridge/" + getParamStr(parameters, "parameters");
    	}
    	if(parameterFilename.size() > 0) {
    		// Load file and parse parameters
    		std::ifstream file(parameterFilename.c_str());
    		if(!file.is_open()) {
    			//throw SIPL::IOException(parameterFilename.c_str(), __LINE__, __FILE__); // --> malloc
    			throw SIPL::IOException(parameterFilename.c_str());
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
					setParameter(parameters, name, value);
    			} else {
    				// parameter with no value
    				setParameter(parameters, line, "true");
    			}
    		}
    		file.close();
    	}
    }
}

paramList initParameters(std::string parameter_dir) {
	paramList parameters;

	std::ifstream file;
//	std::string filename = std::string(PARAMETERS_DIR)+"/parameters";
	std::string filename = parameter_dir+"/parameters";

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
			string description = line.substr(pos+2, line.find("\"", pos+2)-(pos+2));
			string group = line.substr(line.find("\"", pos+2)+2, line.length()-line.find("\"", pos+2)-1);
			BoolParameter v = BoolParameter(defaultValue == "true", description, group);
			parameters.bools[name] = v;
		} else if(type == "num") {
			line = line.substr(pos+1);
			pos = line.find(" ");
			float min = stringToFloat(line.substr(0,pos));
			line = line.substr(pos+1);
			pos = line.find(" ");
			float max = stringToFloat(line.substr(0,pos));
			line = line.substr(pos+1);
			float step = stringToFloat(line);

			int descriptionStart = line.find("\"");
			string description = line.substr(descriptionStart+1, line.find("\"", descriptionStart+1)-(descriptionStart+1));
			string group = line.substr(line.find("\"", descriptionStart+1)+2, line.length()-(line.find("\"", descriptionStart+1)+1));
			NumericParameter v = NumericParameter(stringToFloat(defaultValue), min, max, step, description, group);
			parameters.numerics[name] = v;
		} else if(type == "str") {
			vector<string> list ;
			int descriptionStart = line.find("\"");
			if(descriptionStart-pos > 1) {
				list = split(line.substr(pos+1, descriptionStart-(pos+1)), " ");
			}

			string description = line.substr(descriptionStart+1, line.find("\"", descriptionStart+1)-(descriptionStart+1));
			string group = line.substr(line.find("\"", descriptionStart+1)+2, line.length()-(line.find("\"", descriptionStart+1)+1));
			StringParameter v = StringParameter(defaultValue, list, description, group);
			parameters.strings[name] = v;
		} else {
	    	std::string str = "Could not parse parameter of type: " + std::string(type);
	        throw SIPL::SIPLException(str.c_str());
		}

		getline(file, line);
	}

	return parameters;
}

void setParameter(paramList &parameters, string name, string value) {
	if(parameters.bools.count(name) > 0) {
		BoolParameter v = parameters.bools[name];
		bool boolValue = (value == "true") ? true : false;
		v.set(boolValue);
		parameters.bools[name] = v;
	} else if(parameters.numerics.count(name) > 0) {
		NumericParameter v = parameters.numerics[name];
		v.set(stringToFloat(value));
		parameters.numerics[name] = v;
	} else if(parameters.strings.count(name) > 0) {
		StringParameter v = parameters.strings[name];
		if(name == "parameters") {
		    // Set parameters value without any validation
		    v.setWithoutValidation(value);
		} else {
            v.set(value);
		}
		parameters.strings[name] = v;
	} else {
    	std::string str = "Can not set value for parameter with name: " + name;
        throw SIPL::SIPLException(str.c_str());
	}

}

float getParam(paramList parameters, string parameterName) {
	if(parameters.numerics.count(parameterName) == 0) {
    	std::string str = "numeric parameter not found: " + parameterName;
        throw SIPL::SIPLException(str.c_str());
	}
	NumericParameter v = parameters.numerics[parameterName];
	return v.get();
}

bool getParamBool(paramList parameters, string parameterName) {
	if(parameters.bools.count(parameterName) == 0) {
    	std::string str = "bool parameter not found: " + parameterName;
        throw SIPL::SIPLException(str.c_str());
	}
	BoolParameter v = parameters.bools[parameterName];
	return v.get();
}

string getParamStr(paramList parameters, string parameterName) {
	if(parameters.strings.count(parameterName) == 0) {
    	std::string str = "string parameter not found: " + parameterName;
        throw SIPL::SIPLException(str.c_str());
	}
	StringParameter v = parameters.strings[parameterName];
	return v.get();
}

paramList getParameters(int argc, char ** argv) {
	paramList parameters = initParameters(std::string(PARAMETERS_DIR));

    // Go through each parameter, first parameter is filename
    // Try to see if the parameters parameter and centerline-method is set
	for(int i = 2; i < argc; i++) {
		string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            string nextToken = "";
            if(i+1 < argc) {
                nextToken = argv[i+1];
                if(nextToken.substr(0,2) != "--") {
					i++;
                }
            }
            if(token.substr(2) == "parameters" || token.substr(2) == "centerline-method")
				setParameter(parameters, token.substr(2), nextToken);
        }
    }

    // If a parameter preset is given load these values
    loadParameterPreset(parameters, std::string(PARAMETERS_DIR));

    // Go through each parameter, first parameter is filename
	for(int i = 2; i < argc; i++) {
        string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            string nextToken = "true";
            if(i+1 < argc) {
                nextToken = argv[i+1];
                if(nextToken.substr(0,2) == "--") {
                	nextToken = "true";
                } else {
					i++;
                }
            }
			setParameter(parameters, token.substr(2), nextToken);
        }
    }

	return parameters;
}

BoolParameter::BoolParameter(bool defaultValue, string description, string group) {
	this->value = defaultValue;
	this->description = description;
	this->group = group;
}

bool BoolParameter::get() {
	return this->value;
}

void BoolParameter::set(bool value) {
	this->value = value;
}

NumericParameter::NumericParameter(float defaultValue, float min, float max, float step, string description, string group) {
	this->min = min;
	this->max = max;
	this->step = step;
	this->set(defaultValue);
	this->description = description;
	this->group = group;
}

float NumericParameter::get() {
	return this->value;
}

void NumericParameter::set(float value) {
	if(this->validate(value)) {
		this->value = value;
	} else {
		throw SIPL::SIPLException("Error in setting numerical parameter ", __LINE__, __FILE__);
	}
}

bool NumericParameter::validate(float value) {
	return (value >= min) && (value <= max) ;//&& ((float)ceil((value-min)/step) - (float)(value-min)/step < 0.0001);
}

StringParameter::StringParameter(string defaultValue, vector<string> possibilities, string description, string group) {
	this->possibilities = possibilities;
	this->set(defaultValue);
	this->description = description;
	this->group = group;
}

string StringParameter::get() {
	return this->value;
}

void StringParameter::set(string value) {
	if(this->validate(value)) {
		this->value = value;
	} else {
		throw SIPL::SIPLException("Error in setting string parameter", __LINE__, __FILE__);
	}
}

void StringParameter::setWithoutValidation(std::string value) {
    this->value = value;
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

std::string BoolParameter::getGroup() const {
	return group;
}

void BoolParameter::setGroup(std::string group) {
	this->group = group;
}

std::string NumericParameter::getGroup() const {
	return group;
}

void NumericParameter::setGroup(std::string group) {
	this->group = group;
}

std::string StringParameter::getGroup() const {
	return group;
}


void StringParameter::setGroup(std::string group) {
	this->group = group;
}









