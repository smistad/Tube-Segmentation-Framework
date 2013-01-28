#include <parameters.hpp>
using namespace std;

paramList initParameters() {
	// TODO: get from a file
	paramList parameters;

	std::ifstream file;
	file.open("parameters/parameters");
	string line;
	getline(file, line);
	while(!file.eofbit()) {
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
			BoolParameter v = BoolParameter(defaultValue == "true");
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
			NumericParameter v = NumericParameter(atof(defaultValue.c_str()), min, max, step);
			parameters.numerics[name] = v;
		} else if(type == "str") {

			vector<string> list;
			int end = line.find(" ");
			while(end < line.length()) {
				end = line.find(" ", pos+1);
				list.push_back(line.substr(pos,end));
				pos = end+1;
			}

			StringParameter v = StringParameter(defaultValue, list);
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
	} else if(parameters.numerics.count(name) > 0) {
		NumericParameter v = parameters.numerics[name];
		if(!v.validate(atof(value.c_str())))
			throw exception();
		v.set(atof(value.c_str()));
	} else if(parameters.strings.count(name) > 0) {
		StringParameter v = parameters.strings[name];
		if(!v.validate(value))
			throw exception();
		v.set(value);

	} else {
		throw exception();
	}

	return parameters;
}

float getParam(paramList parameters, string parameterName) {
	NumericParameter v = parameters.numerics[parameterName];
	return v.get();
}

bool getParamBool(paramList parameters, string parameterName) {
	BoolParameter v = parameters.bools[parameterName];
	return v.get();
}

string getParamStr(paramList parameters, string parameterName) {
	StringParameter v = parameters.bools[parameterName];
	return v.get();
}

paramList getParameters(int argc, char ** argv) {
	paramList parameters = initParameters();

    // Go through each parameter, first parameter is filename
    for(int i = 2; i < argc; i++) {
        string token = argv[i];
        if(token.substr(0,2) == "--") {
            // Check to see if the parameter has a value
            string nextToken;
            if(i+1 < argc) {
                nextToken = argv[i+1];
            } else {
                nextToken = "";
            }
			paramList parameters = setParameter(parameters, token.substr(2), nextToken);
			i++;
        }
    }

	return parameters;
}

BoolParameter::BoolParameter(bool defaultValue) {
	this->value = defaultValue;
}

bool BoolParameter::get() {
	return this->value;
}

void BoolParameter::set(bool value) {
	this->value = value;
}

NumericParameter::NumericParameter(float defaultValue, float min, float max, float step) {
	this->value = defaultValue;
	this->min = min;
	this->max = max;
	this->step = step;
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

StringParameter::StringParameter(string defaultValue, vector<string> possibilities) {
	this->value = defaultValue;
	this->possibilities = possibilities;
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
	vector<string>::iterator it;
	bool found = false;
	for(it = possibilities.begin(); it != possibilities.end(); it++) {
		if(value == *it) {
			found = true;
			break;
		}
	}
	return found;
}

