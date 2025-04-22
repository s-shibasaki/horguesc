#include "Config.h"

using namespace System;
using namespace System::IO;

Config::Config() {
	_sid = "UNKNOWN";
	_startYear = 1986;

	_dbHost = "localhost";
	_dbPort = 5432;
	_dbName = "horguesc";
	_dbUsername = "postgres";
	_dbPassword = "postgres";
}

bool Config::Load(String^ iniFile) {
	Console::WriteLine("Loading configuration from: {0}", iniFile);

	if (!File::Exists(iniFile)) {
		Console::WriteLine("File not found: {0}", iniFile);
		return false;
	}

	array<String^>^ lines = File::ReadAllLines(iniFile);

	String^ currentSection = "";
	for each (String ^ line in lines) {
		line = line->Trim();
		if (line->StartsWith(";") || line->Length == 0)
			continue;

		if (line->StartsWith("[") && line->EndsWith("]")) {
			currentSection = line->Substring(1, line->Length - 2);
			continue;
		}

		int equalPos = line->IndexOf('=');
		if (equalPos == -1)
			continue;

		String^ key = line->Substring(0, equalPos)->Trim();
		String^ value = line->Substring(equalPos + 1)->Trim();

		if (currentSection == "dataloader") {
			if (key == "Sid") {
				Console::WriteLine("Sid: {0}", value);
				_sid = value;
			}
			if (key == "StartYear") {
				int intValue = Int32::Parse(value);
				Console::WriteLine("StartYear: {0}", value);
				_startYear = intValue;
			}
		}
		else if (currentSection == "database") {
			if (key == "Host") {
				Console::WriteLine("Database Host: {0}", value);
				_dbHost = value;
			}
			else if (key == "Port") {
				int intValue = Int32::Parse(value);
				Console::WriteLine("Database Port: {0}", intValue);
				_dbPort = intValue;
			}
			else if (key == "Database") {
				Console::WriteLine("Database Name: {0}", value);
				_dbName = value;
			}
			else if (key == "Username") {
				Console::WriteLine("Database Username: {0}", value);
				_dbUsername = value;
			}
			else if (key == "Password") {
				Console::WriteLine("Database Password: ********");
				_dbPassword = value;
			}

		}
	}

	return true;
}