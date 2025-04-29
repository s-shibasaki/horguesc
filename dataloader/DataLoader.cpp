#include "DataLoader.h"
#include "Config.h"
#include "ProgressBar.h"
#include "RecordProcessor.h"

using namespace System;
using namespace AxJVDTLabLib;
using namespace System::Runtime::InteropServices;
using namespace Npgsql;

DataLoader::DataLoader(AxJVLink^ jvlink) {
	this->jvlink = jvlink;
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

bool DataLoader::InitializeDatabase() {
	try {
		Console::Write("Connecting to postgres: ");
		String^ connectionString = String::Format("Host={0};Port={1};Database=postgres;Username={2};Password={3};", config->DbHost, config->DbPort, config->DbUsername, config->DbPassword);
		NpgsqlConnection^ connection = gcnew NpgsqlConnection(connectionString);
		connection->Open();
		Console::WriteLine("OK.");

		// 共有する変数
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
		Object^ result;

		Console::Write("Checking if database exists: ");
		command->CommandText = "SELECT 1 FROM pg_database WHERE datname = @dbname";
		command->Parameters->Clear();
		command->Parameters->AddWithValue("@dbname", config->DbName);
		result = command->ExecuteScalar();

		if (result != nullptr) {
			// データベースが存在した
			Console::WriteLine("exists.");

			// データベースを削除しない場合、このまま抜ける
			if (!config->DeleteDatabase)
				return true;

			// 削除する場合
			Console::Write("Deleting database: ");
			command->CommandText = String::Format("DROP DATABASE \"{0}\"", config->DbName);
			command->Parameters->Clear();
			command->ExecuteNonQuery();
			Console::WriteLine("OK.");
		}
		else {
			// データベースが存在しなかった
			Console::WriteLine("does not exists.");
		}

		// データベース削除後、または存在しなかった場合
		Console::Write("Creating new database: ");
		command->CommandText = String::Format("CREATE DATABASE \"{0}\"", config->DbName);
		command->Parameters->Clear();
		command->ExecuteNonQuery();
		Console::WriteLine("OK.");
		return true;
	}
	catch (Exception^ ex) {
		Console::WriteLine("failed. {0}", ex->Message);
		return false;
	}
}

bool DataLoader::Initialize() {
	config = gcnew Config();
	config->Load("horguesc.ini");

	if (!InitializeDatabase()) {
		Console::WriteLine("Database initialization failed.");
		return false;
	}

	Console::Write("Opening database connection: ");
	try {
		String^ connectionString = String::Format("Host={0};Port={1};Database={2};Username={3};Password={4};", config->DbHost, config->DbPort, config->DbName, config->DbUsername, config->DbPassword);
		connection = gcnew NpgsqlConnection(connectionString);
		connection->Open();
	}
	catch (Exception^ ex) {
		Console::WriteLine("failed. {0}", ex->Message);
		return false;
	}
	Console::WriteLine("OK.");

	recordProcessor = gcnew RecordProcessor(connection);
	if (!recordProcessor->Initialize()) {
		Console::WriteLine("Failed to initialize Record Processor.");
		return false;
	}

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

	int downloadErrorCode = WaitForDownloadCompletion(downloadCount);
	if (downloadErrorCode != 0) {
		Console::WriteLine("Error occurred during file download. Error code: {0}", downloadErrorCode);
		return false;
	}


	ProgressBar^ jvGetsProgressBar = gcnew ProgressBar(readCount, 50, "Reading files");
	try {
		int jvReadReturnCode;
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

			// Convert the string to CP932 byte array
			array<Byte>^ bytes = System::Text::Encoding::GetEncoding(932)->GetBytes(buffer);

			// Process the record
			int processResult = recordProcessor->ProcessRecord(bytes);

			if (processResult == RecordProcessor::PROCESS_ERROR) {
				// RecordProcessorで改行済み
				Console::WriteLine("Error occured during process record.");
				return false;
			}
			else if (processResult == RecordProcessor::PROCESS_SKIP) {
				jvlink->JVSkip();
				jvGetsProgressBar->Increment();
			}
		}
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error occured during data read: {0}", ex->Message);
		return false;
	}

	jvlink->JVClose();

	return true;
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

bool DataLoader::RollbackTransaction(NpgsqlTransaction^ transaction) {
	if (transaction != nullptr) {
		Console::Write("Attempting to rollback: ");
		try {
			transaction->Rollback();
			Console::WriteLine("OK.");
		}
		catch (Exception^ rollbackEx) {
			Console::WriteLine("failed: {0}", rollbackEx->Message);
			return false;
		}
	}
	return true;
}