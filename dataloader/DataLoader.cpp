#include "DataLoader.h"
#include "Config.h"
#include "ProgressBar.h"
#include "RecordProcessor.h"

using namespace System;
using namespace AxJVDTLabLib;
using namespace System::Runtime::InteropServices;
using namespace Npgsql;

DataLoader::DataLoader(AxJVLink^ jvlink, array<String^>^ args) {
	this->jvlink = jvlink;
	this->args = args;
}

bool DataLoader::Execute() {
	if (args->Length == 0) {
		Console::WriteLine("No arguments provided.");
		return false;
	}
	if (args[0] == "setup") {
		return ExecuteSetup();
	}
	else if (args[0] == "update") {
		return ExecuteUpdate();
	}
	else if (args[0] == "realtime") {
		return ExecuteRealtime();
	}
	else {
		Console::WriteLine("Invalid argument: {0}", args[0]);
		return false;
	}
}

bool DataLoader::ExecuteSetup() {
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

bool DataLoader::ExecuteUpdate() {
    if (!Initialize()) {
        Console::WriteLine("Initialization failed.");
        return false;
    }

    // Get the last timestamp from the database
    Console::Write("Getting last timestamp from database: ");
    String^ lastFromDate = nullptr;
    try {
        NpgsqlCommand^ command = gcnew NpgsqlCommand(
            "SELECT timestamp FROM last_file_timestamp WHERE id = 1", 
            connection);
        Object^ result = command->ExecuteScalar();
        
        if (result != nullptr && result != DBNull::Value) {
            lastFromDate = result->ToString();
            Console::WriteLine("OK. Last timestamp: {0}", lastFromDate);
        } else {
            Console::WriteLine("failed.");
            Console::WriteLine("No timestamp found in database. Please run setup first.");
            Console::WriteLine("Run the program with 'setup' argument to initialize the database.");
            return false;
        }
    } catch (Exception^ ex) {
        Console::WriteLine("failed. {0}", ex->Message);
        Console::WriteLine("No timestamp found in database. Please run setup first.");
        Console::WriteLine("Run the program with 'setup' argument to initialize the database.");
        return false;
    }

    JVOpenParams^ lastParams = gcnew JVOpenParams("RACEBLDNSNPNSLOPWOODYSCHMINGTOKUDIFNHOSNHOYUCOMM", lastFromDate, 1);
    if (!ProcessChunk(lastParams)) {
        Console::WriteLine("Error occured during data loading process.");
        return false;
    }

    return true;
}

bool DataLoader::ExecuteRealtime() {
    if (!Initialize()) {
        Console::WriteLine("Initialization failed.");
        return false;
    }

    // Today's date in YYYYMMDD format
    String^ today = DateTime::Now.ToString("yyyyMMdd");
	//today = "20250518"; // For testing purposes, set a specific date
    
    // Process date-based data specs first
    array<String^>^ dateBasedDataSpecs = {"0B15", "0B11", "0B14", "0B13", "0B17", "0B51"};
    for each (String^ dataSpec in dateBasedDataSpecs) {
        if (!ProcessRealtimeChunk(dataSpec, today)) {
            Console::WriteLine("Error occurred during realtime data processing for {0}.", dataSpec);
            return false;
        }
    }
    
    // Now process race ID-based data specs (0B30, 0B20)
    try {
        // Get today's race IDs from the database
        NpgsqlCommand^ command = gcnew NpgsqlCommand(
            "SELECT CONCAT(TO_CHAR(kaisai_date, 'YYYYMMDD'), keibajo_code, LPAD(kaisai_kai::text, 2, '0'), " +
            "LPAD(kaisai_nichime::text, 2, '0'), LPAD(kyoso_bango::text, 2, '0')) AS race_id " +
            "FROM ra WHERE kaisai_date = @today::date", 
            connection);
        command->Parameters->AddWithValue("@today", today);
        
        NpgsqlDataReader^ reader = command->ExecuteReader();
        
        // Store the race IDs
        System::Collections::Generic::List<String^>^ raceIds = gcnew System::Collections::Generic::List<String^>();
        while (reader->Read()) {
            raceIds->Add(reader["race_id"]->ToString());
        }
        reader->Close();
        
        // Process each race ID for the race-specific data specs
        //array<String^>^ raceBasedDataSpecs = { "0B30", "0B20", "0B41", "0B42" };
        array<String^>^ raceBasedDataSpecs = { "0B30", "0B20" };
        for each (String^ dataSpec in raceBasedDataSpecs) {
            for each (String^ raceId in raceIds) {
                if (!ProcessRealtimeChunk(dataSpec, raceId)) {
                    Console::WriteLine("Error occurred during realtime data processing for {0} with race ID {1}.", dataSpec, raceId);
                    return false;
                }
            }
        }
    }
    catch (Exception^ ex) {
        Console::WriteLine("Error getting race IDs: {0}", ex->Message);
        return false;
    }
    
    return true;
}

bool DataLoader::ProcessRealtimeChunk(String^ dataSpec, String^ key) {
    // Determine if we're using a date or race ID based on the key length
    bool isRaceId = (key->Length > 8);
    String^ keyType = isRaceId ? "race ID" : "date";
    
    Console::Write("JVRTOpen for {0} with {1} {2}: ", dataSpec, keyType, key);
    int jvRTOpenReturnCode = jvlink->JVRTOpen(dataSpec, key);
    
    if (jvRTOpenReturnCode == -1) {
        Console::WriteLine("No matching data found.");
        return true; // No data is not an error, so return success
    }
    else if (jvRTOpenReturnCode != 0) {
        Console::WriteLine("failed. Error code: {0}", jvRTOpenReturnCode);
        return false;
    }
    Console::WriteLine("OK.");
    
    // Begin transaction
    NpgsqlTransaction^ transaction = nullptr;
    
    try {
        transaction = connection->BeginTransaction();
        
        int jvReadReturnCode;
        while (true) {
            String^ fileName;
            int size = 110000;
            String^ buffer;
            jvReadReturnCode = jvlink->JVRead(buffer, size, fileName);
            
            if (jvReadReturnCode < -1) {
                Console::WriteLine("Error occurred during data read. Error code: {0}", jvReadReturnCode);
                RollbackTransaction(transaction);
                return false;
            }
            else if (jvReadReturnCode == -1) {
                continue;
            }
            else if (jvReadReturnCode == 0) {
                break;
            }
            
            // Convert the string to CP932 byte array
            array<Byte>^ bytes = System::Text::Encoding::GetEncoding(932)->GetBytes(buffer);
            
            // Process the record
            int processResult = recordProcessor->ProcessRecord(bytes);
            
            if (processResult == RecordProcessor::PROCESS_ERROR) {
                Console::WriteLine("Error occurred during process record.");
                RollbackTransaction(transaction);
                return false;
            }
            else if (processResult == RecordProcessor::PROCESS_SKIP) {
                jvlink->JVSkip();
            }
        }
        
        // Commit transaction
        Console::Write("Committing transaction: ");
        transaction->Commit();
        Console::WriteLine("OK.");
    }
    catch (Exception^ ex) {
        Console::WriteLine("Error occurred during data read: {0}", ex->Message);
        RollbackTransaction(transaction);
        return false;
    }
    finally {
        jvlink->JVClose();
    }
    
    return true;
}


bool DataLoader::InitializeDatabase(bool deleteIfExists) {
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
            if (!deleteIfExists)
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
    config->Load();

    // Only allow database deletion in setup mode
    bool shouldDeleteDatabase = config->DeleteDatabase && (args->Length > 0 && args[0] == "setup");
    
    if (!InitializeDatabase(shouldDeleteDatabase)) {
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

    // Create timestamp table in the database with transaction
    NpgsqlTransaction^ transaction = nullptr;
    try {
        transaction = connection->BeginTransaction();
        
        recordProcessor = gcnew RecordProcessor(connection);
        if (!recordProcessor->Initialize()) {
            Console::WriteLine("Failed to initialize Record Processor.");
            RollbackTransaction(transaction);
            return false;
        }
        
        if (!CreateTimestampTable()) {
            Console::WriteLine("Failed to create timestamp table.");
            RollbackTransaction(transaction);
            return false;
        }
        
        // Commit the transaction
        transaction->Commit();
    }
    catch (Exception^ ex) {
        Console::WriteLine("Initialization error: {0}", ex->Message);
        RollbackTransaction(transaction);
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
    if (jvOpenReturnCode == -1) {
        Console::WriteLine("No matching data found.");
        return true; // No data is not an error, so return success
    }
    else if (jvOpenReturnCode != 0) {
        Console::WriteLine("failed. Error code: {0}", jvOpenReturnCode);
        return false;
    }
    Console::WriteLine("OK. readCount={0}, downloadCount={1}, lastFileTimestamp={2}", readCount, downloadCount, lastFileTimestamp);

    int downloadErrorCode = WaitForDownloadCompletion(downloadCount);
    if (downloadErrorCode != 0) {
        Console::WriteLine("Error occurred during file download. Error code: {0}", downloadErrorCode);
        return false;
    }

    // Begin transaction for all database operations in this chunk
    NpgsqlTransaction^ transaction = nullptr;
    try {
        transaction = connection->BeginTransaction();
        
        ProgressBar^ jvGetsProgressBar = gcnew ProgressBar(readCount, 50, "Reading files");
        
        int jvReadReturnCode;
        while (true) {
            String^ fileName;
            int size = 110000;
            String^ buffer;
            jvReadReturnCode = jvlink->JVRead(buffer, size, fileName);

            if (jvReadReturnCode < -1) {
                Console::WriteLine();
                Console::WriteLine("Error occurred during data read. Error code: {0}", jvReadReturnCode);
                RollbackTransaction(transaction);
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
                Console::WriteLine("Error occurred during process record.");
                RollbackTransaction(transaction);
                return false;
            }
            else if (processResult == RecordProcessor::PROCESS_SKIP) {
                jvlink->JVSkip();
                jvGetsProgressBar->Increment();
            }
        }

        // Save the lastFileTimestamp to the database after all data is processed
        Console::Write("Updating last file timestamp: ");
        if (UpdateLastFileTimestamp(lastFileTimestamp)) {
            Console::WriteLine("OK.");
        } else {
            Console::WriteLine("Failed.");
            RollbackTransaction(transaction);
            return false;
        }

        // Commit the transaction if everything was successful
        Console::Write("Committing transaction: ");
        transaction->Commit();
        Console::WriteLine("OK.");
        
    }
    catch (Exception^ ex) {
        Console::WriteLine();
        Console::WriteLine("Error occurred during data processing: {0}", ex->Message);
        RollbackTransaction(transaction);
        return false;
    }
    finally {
        jvlink->JVClose();
    }

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

bool DataLoader::CreateTimestampTable() {
    try {
        Console::Write("Creating timestamp table: ");
        NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
        
        command->CommandText =
            "CREATE TABLE IF NOT EXISTS last_file_timestamp ("
            "id SMALLINT PRIMARY KEY DEFAULT 1, "
            "timestamp VARCHAR(255) NOT NULL, "
            "updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "CONSTRAINT single_row CHECK (id = 1))";
        command->ExecuteNonQuery();
        
        Console::WriteLine("OK.");
        return true;
    }
    catch (Exception^ ex) {
        Console::WriteLine("failed. {0}", ex->Message);
        return false;
    }
}

bool DataLoader::UpdateLastFileTimestamp(System::String^ timestamp) {
    try {
        NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
        
        // Use upsert (INSERT ... ON CONFLICT) to handle both insert and update cases
        command->CommandText = 
            "INSERT INTO last_file_timestamp (id, timestamp, updated_at) "
            "VALUES (1, @timestamp, CURRENT_TIMESTAMP) "
            "ON CONFLICT (id) DO UPDATE "
            "SET timestamp = @timestamp, updated_at = CURRENT_TIMESTAMP";
        
        command->Parameters->AddWithValue("@timestamp", timestamp);
        command->ExecuteNonQuery();
        
        return true;
    }
    catch (Exception^ ex) {
        Console::WriteLine("Failed to update last file timestamp: {0}", ex->Message);
        return false;
    }
}