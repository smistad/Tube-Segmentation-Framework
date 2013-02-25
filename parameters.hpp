#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <string>
#include <vector>
#ifdef CPP11
#include <unordered_map>
#include <tuple>
using std::unordered_map;
using std::tuple;
#else
#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
using boost::unordered_map;
using boost::tuple;
#endif



class BoolParameter {
public:
	BoolParameter() {};
	BoolParameter(bool defaultValue, std::string description, std::string group);
	bool get();
	void set(bool value);
	std::string getDescription() const;
	std::string getGroup() const;
	void setGroup(std::string group);
private:
	bool value;
	std::string description;
	std::string group;
};

class NumericParameter {
public:
	NumericParameter() {};
	NumericParameter(float defaultValue, float min, float max, float step, std::string description, std::string group);
	float get();
	void set(float value);
	bool validate(float value);
	float getMax() const;
	void setMax(float max);
	float getMin() const;
	void setMin(float min);
	float getStep() const;
	void setStep(float step);
	std::string getDescription() const;
	std::string getGroup() const;
	void setGroup(std::string group);
private:
	float value;
	float min;
	float max;
	float step;
	std::string description;
	std::string group;
};

class StringParameter {
public:
	StringParameter() {};
	StringParameter(std::string defaultValue, std::vector<std::string> possibilities, std::string description, std::string group);
	std::string get();
	void set(std::string value);
	bool validate(std::string value);
	std::vector<std::string> getPossibilities() const;
	std::string getDescription() const;
	std::string getGroup() const;
	void setGroup(std::string group);
private:
	std::string value;
	std::vector<std::string> possibilities;
	std::string description;
	std::string group;
};
typedef struct paramList {
	unordered_map<std::string, BoolParameter> bools;
	unordered_map<std::string, NumericParameter> numerics;
	unordered_map<std::string, StringParameter> strings;
} paramList;

void loadParameterPreset(paramList &parameters, std::string parameter_dir);
paramList initParameters(std::string parameter_dir);
void setParameter(paramList &parameters, std::string name, std::string value);
paramList getParameters(int argc, char ** argv);
float getParam(paramList parameters, std::string parameterName);
bool getParamBool(paramList parameters, std::string parameterName);
std::string getParamStr(paramList parameters, std::string parameterName);

#endif /* PARAMETERS_HPP_ */
