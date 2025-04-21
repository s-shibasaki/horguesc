#include "Config.h"

using namespace System;

Config::Config() {
	Sid = "UNKNOWN";
}

Config^ Config::LoadFromFile(String^ filePath) {
	Config^ config = gcnew Config();
	return config;
}