#pragma once
ref class Config
{
private:
	System::String^ _sid;
	int _startYear;

public:
	property System::String^ Sid {System::String^ get() { return _sid; }};
	property int StartYear {int get() { return _startYear; }};

	Config();
	bool Load(System::String^ iniFile);
};

