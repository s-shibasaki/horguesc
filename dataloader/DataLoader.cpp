#include "DataLoader.h"
#include "Config.h"

using namespace System;
using namespace AxJVDTLabLib;

DataLoader::DataLoader(AxJVLink^ jvlink) {
	jvlink = jvlink;
}

bool DataLoader::Initialize() {
	config = gcnew Config();
	config->Load("horguesc.ini");



	return true;
}

bool DataLoader::Execute() {
	if (!Initialize()) {
		Console::WriteLine("Initialization failed.");
		return false;
	}

	Console::WriteLine("Sid: {0}", config->Sid);
	Console::WriteLine("StartYear: {0}", config->StartYear);
	return true;
}