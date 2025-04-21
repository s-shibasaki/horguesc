#include "DataLoader.h"
#include "Config.h"
#include "ProgressBar.h"

using namespace System;
using namespace AxJVDTLabLib;

DataLoader::DataLoader(AxJVLink^ jvlink) {
	this->jvlink = jvlink;
}

bool DataLoader::Initialize() {
	config = gcnew Config();
	config->Load("horguesc.ini");

	Console::Write("JVLink Initialization: ");
	int jvInitReturnCode = jvlink->JVInit(config->Sid);
	if (jvInitReturnCode != 0) {
		Console::WriteLine("failed. Error code: {0}", jvInitReturnCode);
		return false;
	}
	Console::WriteLine("OK.");

	return true;
}

bool DataLoader::ProcessChunk(JVOpenParams^ params) {
	int readCount;
	int downloadCount;
	String^ lastFileTimestamp;

	Console::Write("JVOpen: ");
	int jvOpenReturnCode = jvlink->JVOpen(params->dataSpec, params->fromDate, params->option, readCount, downloadCount, lastFileTimestamp);
	if (jvOpenReturnCode != 0) {
		Console::WriteLine("failed. Error code: {0}", jvOpenReturnCode);
		return false;
	}
	Console::WriteLine("OK. readCount={0}, downloadCount={1}, lastFileTimestamp={2}", readCount, downloadCount, lastFileTimestamp);

	if (downloadCount > 0) {
		int jvStatusReturnCode = 0;
		ProgressBar^ jvStatusProgressBar = gcnew ProgressBar(downloadCount, 50, "Downloading files");
		while (jvStatusReturnCode != downloadCount) {
			jvStatusReturnCode = jvlink->JVStatus();
			if (jvStatusReturnCode < 0) {
				jvStatusProgressBar->Abort();
				Console::WriteLine("Error occurred during file download. Error code: {0}", jvStatusReturnCode);
				return false;
			}
			jvStatusProgressBar->Update(jvStatusReturnCode);
		}
		jvStatusProgressBar->Complete();
	}

	jvlink->JVClose();

	return true;
}

bool DataLoader::Execute() {
	if (!Initialize()) {
		Console::WriteLine("Initialization failed.");
		return false;
	}

	int currentYear = DateTime::Now.Year;
	for (int year = config->StartYear; year < currentYear; year++) {
		String^ fromDate = String::Format("{0}0101000000-{1}0101000000", year, year + 1);
		JVOpenParams^ params = gcnew JVOpenParams("RACEBLDNSNPNSLOPWOODYSCHMING", fromDate, 4);
		if (!ProcessChunk(params)) {
			Console::WriteLine("Data loading process for {0} failed.", year);
			return false;
		}
	}
	String^ lastFromDate = String::Format("{0}0101000000", currentYear);
	JVOpenParams^ lastParams = gcnew JVOpenParams("RACEBLDNSNPNSLOPWOODYSCHMINGTOKUDIFNHOSNHOYUCOMM", lastFromDate, 4);
	if (!ProcessChunk(lastParams)) {
		Console::WriteLine("Data loading process for {0} failed.", currentYear);
		return false;
	}

	Console::WriteLine("Sid: {0}", config->Sid);
	Console::WriteLine("StartYear: {0}", config->StartYear);
	return true;
}