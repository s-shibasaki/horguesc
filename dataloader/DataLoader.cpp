#include "DataLoader.h"
#include "Config.h"

using namespace System;
using namespace AxJVDTLabLib;

DataLoader::DataLoader(AxJVLink^ jvlink) {
	m_jvlink = jvlink;
}

bool DataLoader::Initialize() {
	m_config = Config::LoadFromFile("config.json");
	return true;
}

bool DataLoader::Execute() {
	if (!Initialize()) {
		Console::WriteLine("Initialization failed.");
		return false;
	}

	Console::WriteLine("SID: {0}", m_config->Sid);
	return true;
}