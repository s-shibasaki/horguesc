#include "DataLoader.h"
#include "Config.h"
#include "ProgressBar.h"

using namespace System;
using namespace AxJVDTLabLib;
using namespace System::Runtime::InteropServices;

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

bool DataLoader::Execute() {
	if (!Initialize()) {
		Console::WriteLine("Initialization failed.");
		return false;
	}

	int currentYear = DateTime::Now.Year;
	for (int year = config->StartYear; year < currentYear; year++) {
		Console::WriteLine("Processing data for year: {0}", year);
		String^ fromDate = String::Format("{0}0101000000-{1}0101000000", year, year + 1);
		JVOpenParams^ params = gcnew JVOpenParams("RACEBLDNSNPNSLOPWOODYSCHMING", fromDate, 4);
		if (!ProcessChunk(params)) {
			Console::WriteLine("Data loading process for {0} failed.", year);
			return false;
		}
	}
	Console::WriteLine("Processing data for year: {0}", currentYear);
	String^ lastFromDate = String::Format("{0}0101000000", currentYear);
	JVOpenParams^ lastParams = gcnew JVOpenParams("RACEBLDNSNPNSLOPWOODYSCHMINGTOKUDIFNHOSNHOYUCOMM", lastFromDate, 4);
	if (!ProcessChunk(lastParams)) {
		Console::WriteLine("Data loading process for {0} failed.", currentYear);
		return false;
	}

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

	int downloadErrorCode = WaitForDownloadCompletion(downloadCount);
	if (downloadErrorCode != 0) {
		Console::WriteLine("Error occurred during file download. Error code: {0}", downloadErrorCode);
		return false;
	}

	int jvReadReturnCode;
	ProgressBar^ jvGetsProgressBar = gcnew ProgressBar(readCount, 50, "Reading files");
	while (true) {
		String^ fileName;
		int size = 110000;
		String^ buffer;
		jvReadReturnCode = jvlink->JVRead(buffer, size, fileName);

		if (jvReadReturnCode < -1) {
			Console::WriteLine();
			Console::WriteLine("Error occured during data read. Error code: {0}", jvReadReturnCode);
			return false;
		}
		else if (jvReadReturnCode == -1) {
			jvGetsProgressBar->Increment();
			continue;
		}
		else if (jvReadReturnCode == 0) {
			jvGetsProgressBar->Increment();
			Console::WriteLine();
			break;
		}
		if (!ProcessRecord(buffer)) {
			Console::WriteLine();
			Console::WriteLine("Error occured during process record.");
			return false;
		}
	}

	jvlink->JVClose();

	return true;
}

bool DataLoader::ProcessRecord(String^ line) {
	return false;
}

int DataLoader::WaitForDownloadCompletion(int downloadCount) {
	if (downloadCount == 0)
		return 0;
	int jvStatusReturnCode = 0;
	ProgressBar^ jvStatusProgressBar = gcnew ProgressBar(downloadCount, 50, "Downloading files");
	while (jvStatusReturnCode != downloadCount) {
		jvStatusReturnCode = jvlink->JVStatus();
		if (jvStatusReturnCode < 0) {
			Console::WriteLine();
			return jvStatusReturnCode;
		}
		jvStatusProgressBar->Update(jvStatusReturnCode);
	}
	Console::WriteLine();
	return 0;
}

